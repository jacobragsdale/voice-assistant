import pyaudio
import wave
import tempfile
import json
import time
import os
import sys
from vosk import Model, KaldiRecognizer, SetLogLevel

# Silence Vosk logs by setting log level to -1 (disable)
SetLogLevel(-1)

class VoiceListener:
    """Handles voice recording, wake word detection, and transcription."""
    
    def __init__(self, model_path, wake_word, sample_rate=16000, buffer_size=1024):
        """
        Initialize a new VoiceListener.
        
        Args:
            model_path (str): Path to the Vosk model
            wake_word (str): Word to trigger the assistant
            sample_rate (int): Audio sample rate
            buffer_size (int): Audio buffer size
        """
        self.model_path = model_path
        self.wake_word = wake_word.lower()
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
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
    
    def record_audio(self, seconds=5):
        """
        Record audio from microphone for specified number of seconds.
        
        Args:
            seconds (int): Duration to record in seconds
            
        Returns:
            str: Path to the recorded audio file
        """
        # Audio recording parameters
        CHUNK = self.buffer_size
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = self.sample_rate
        
        p = pyaudio.PyAudio()
        
        print(f"Recording for {seconds} seconds...")
        
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        
        # Record for the specified duration
        for i in range(0, int(RATE / CHUNK * seconds)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except OSError as e:
                print(f"Warning: {e}")
        
        print("Recording finished.")
        
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
    
    def transcribe_audio(self, audio_file_path):
        """
        Transcribe audio file using local Vosk model.
        
        Args:
            audio_file_path (str): Path to the audio file to transcribe
            
        Returns:
            str or None: Transcribed text or None if transcription failed
        """
        try:
            model = Model(self.model_path)
            with wave.open(audio_file_path, "rb") as wf:
                # Check if the audio format matches what Vosk expects
                if (
                    wf.getnchannels() != 1
                    or wf.getsampwidth() != 2
                    or wf.getcomptype() != "NONE"
                ):
                    print("Audio file must be WAV format mono PCM.")
                    return None

                # Create recognizer
                rec = KaldiRecognizer(model, wf.getframerate())
                rec.SetWords(True)

                # Process audio in chunks
                results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        part_result = json.loads(rec.Result())
                        results.append(part_result.get("text", ""))

                # Get final result
                part_result = json.loads(rec.FinalResult())
                results.append(part_result.get("text", ""))

                # Join all results
                return " ".join(results)
            
        except Exception as e:
            print(f"Error during Vosk transcription: {e}")
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