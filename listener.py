import pyaudio
import wave
import tempfile
import json
import time
import os
from collections import deque

import numpy as np
import webrtcvad
import noisereduce as nr
from openai import OpenAI
from vosk import Model, KaldiRecognizer, SetLogLevel

# Silence Vosk logs by setting log level to -1 (disable)
SetLogLevel(-1)

class VoiceListener:
    """Handles voice recording, wake word detection, and transcription."""

    def __init__(self, model_path, wake_word, sample_rate=16000, buffer_size=1024,
                 vad_aggressiveness=2, silence_duration=1.5, max_record_seconds=15):
        """
        Initialize a new VoiceListener.

        Args:
            model_path (str): Path to the Vosk model
            wake_word (str): Word to trigger the assistant
            sample_rate (int): Audio sample rate
            buffer_size (int): Audio buffer size
            vad_aggressiveness (int): VAD aggressiveness (0-3), higher = more aggressive filtering
            silence_duration (float): Seconds of silence before stopping recording
            max_record_seconds (int): Maximum recording duration in seconds
        """
        self.model_path = model_path
        self.wake_word = wake_word.lower()
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.silence_duration = silence_duration
        self.max_record_seconds = max_record_seconds

        # VAD setup
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.vad_frame_duration_ms = 30
        self.vad_frame_size = int(self.sample_rate * self.vad_frame_duration_ms / 1000)  # 480

        # OpenAI client for Whisper transcription
        self.openai_client = OpenAI()

        # Ensure Vosk logging is disabled
        SetLogLevel(-1)

    def listen_for_wake_word(self):
        """
        Listen for wake word using Vosk model.

        Returns:
            bool: True if wake word was detected, False otherwise
        """
        model = Model(self.model_path)

        # Setup PyAudio input stream
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.buffer_size)

        # Create Vosk recognizer
        rec = KaldiRecognizer(model, self.sample_rate)
        rec.SetWords(True)

        print(f"Listening for wake word: '{self.wake_word}'...")

        # Start listening
        try:
            while True:
                try:
                    data = stream.read(self.buffer_size, exception_on_overflow=False)
                    if len(data) == 0:
                        break

                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())

                        if 'text' in result:
                            text = result['text'].lower()
                            print(f"Heard: {text}")

                            # Check if wake word is in the recognized text
                            if self.wake_word in text:
                                print(f"Wake word detected! '{self.wake_word}'")
                                stream.stop_stream()
                                stream.close()
                                p.terminate()
                                return True
                except OSError as e:
                    print(f"Warning: {e}")
                    time.sleep(0.1)  # Give the buffer a moment to clear
                    continue
        except Exception as e:
            print(f"Error in wake word detection: {e}")

        stream.stop_stream()
        stream.close()
        p.terminate()
        return False

    def record_audio(self):
        """
        Record audio from microphone using VAD to detect speech boundaries.
        Recording starts capturing when speech is detected and stops after
        a period of silence.

        Returns:
            str: Path to the recorded audio file
        """
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = self.sample_rate
        CHUNK = self.vad_frame_size  # 480 samples = 30ms at 16kHz

        p = pyaudio.PyAudio()

        print("Listening for speech (VAD active)...")

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        speech_detected = False
        silent_frame_count = 0
        total_frame_count = 0

        # Pre-speech ring buffer to capture audio just before speech starts (~300ms)
        pre_speech_buffer = deque(maxlen=10)

        # Calculate thresholds
        silence_threshold = int(self.silence_duration / (self.vad_frame_duration_ms / 1000))
        max_frames = int(self.max_record_seconds / (self.vad_frame_duration_ms / 1000))

        try:
            while True:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    total_frame_count += 1
                except OSError as e:
                    print(f"Warning: {e}")
                    continue

                # Check if this frame contains speech
                is_speech = self.vad.is_speech(data, RATE)

                if not speech_detected:
                    # Haven't detected speech yet - buffer frames
                    pre_speech_buffer.append(data)

                    if is_speech:
                        # Speech started! Flush pre-speech buffer
                        speech_detected = True
                        print("Speech detected, recording...")
                        frames.extend(pre_speech_buffer)
                        silent_frame_count = 0
                    elif total_frame_count >= max_frames:
                        # Timeout waiting for speech
                        print("No speech was detected.")
                        break
                else:
                    # Currently recording speech
                    frames.append(data)

                    if is_speech:
                        silent_frame_count = 0
                    else:
                        silent_frame_count += 1

                    # Stop if silence exceeds threshold
                    if silent_frame_count >= silence_threshold:
                        duration = len(frames) * self.vad_frame_duration_ms / 1000
                        print(f"Recording finished. Duration: {duration:.1f}s")
                        break

                    # Safety cap on recording duration
                    if len(frames) >= max_frames:
                        print(f"Recording reached maximum duration ({self.max_record_seconds}s).")
                        break

        except Exception as e:
            print(f"Error during recording: {e}")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save recording to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_filename = temp_audio_file.name

        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        return temp_filename

    def reduce_noise(self, audio_file_path):
        """
        Apply noise reduction to an audio file.

        Args:
            audio_file_path (str): Path to the WAV audio file

        Returns:
            str: Path to the processed audio file (same path, overwritten)
        """
        try:
            wf = wave.open(audio_file_path, 'rb')
            framerate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw_data = wf.readframes(wf.getnframes())
            wf.close()

            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)

            if len(audio_data) == 0:
                return audio_file_path

            reduced = nr.reduce_noise(y=audio_data, sr=framerate, stationary=True, prop_decrease=0.75)
            reduced_int16 = reduced.astype(np.int16)

            wf = wave.open(audio_file_path, 'wb')
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(reduced_int16.tobytes())
            wf.close()

            print("Noise reduction applied.")
            return audio_file_path
        except Exception as e:
            print(f"Warning: Noise reduction failed ({e}), using original audio.")
            return audio_file_path

    def transcribe_audio(self, audio_file_path):
        """
        Transcribe audio file using OpenAI Whisper API.

        Args:
            audio_file_path (str): Path to the audio file to transcribe

        Returns:
            str or None: Transcribed text or None if transcription failed
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            text = response.text.strip()
            return text if text else None
        except Exception as e:
            print(f"Error during Whisper transcription: {e}")
            return None

    def cleanup_audio_file(self, file_path):
        """
        Delete a temporary audio file.

        Args:
            file_path (str): Path to the file to delete
        """
        try:
            os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Failed to delete temporary file {file_path}: {e}")
            pass
