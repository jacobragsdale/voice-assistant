import warnings
import sounddevice as sd
from kokoro import KPipeline
import threading


class VoiceAssistant:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VoiceAssistant, cls).__new__(cls)
        return cls._instance

    def __init__(self, lang_code='a', default_voice='af_heart', samplerate=24000):
        if VoiceAssistant._initialized:
            return
        # Pass repo_id to suppress the repo_id warning from KPipeline.
        self.pipeline = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')
        self.default_voice = default_voice
        self.samplerate = samplerate
        self.threads = []
        VoiceAssistant._initialized = True

    def _speak_thread(self, text, voice):
        # Use the provided voice or fallback to the default.
        voice_to_use = voice if voice is not None else self.default_voice
        generator = self.pipeline(text, voice=voice_to_use)
        # Iterate over the segments and play each audio segment in real time.
        for idx, (_, _, audio) in enumerate(generator):
            sd.play(audio, samplerate=self.samplerate)
            sd.wait()

    def speak(self, text, voice=None, threaded=True):
        if threaded:
            for thread in self.threads:
                thread.join()
            self.threads.clear()

            thread = threading.Thread(target=self._speak_thread, args=(text, voice))
            thread.start()
            self.threads.append(thread)
        else:
            self._speak_thread(text, voice)

if __name__ == '__main__':
    assistant = VoiceAssistant()

    # List of wake responses to try out.
    wake_responses = ["Ya", "How can I help?", "Sup", "Hey", "Huh?"]

    for response in wake_responses:
        assistant.speak(response)
        for thread in assistant.threads:
            thread.join()
        assistant.threads.clear()
