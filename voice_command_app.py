import os
import sys
from dotenv import load_dotenv
from download_model import download_model
from commands.registry import create_command_registry
from listener import VoiceListener
from command_processor import CommandProcessor
from tts import VoiceAssistant

# Configuration
WAKE_WORD = "computer"  # The keyword to trigger recording
COMMAND_TIMEOUT = 5  # Recording duration for commands in seconds
SAMPLE_RATE = 16000  # Audio sample rate
BUFFER_SIZE = 1024  # Buffer size for audio stream

assistant = VoiceAssistant()

def main():
    """Main application entry point."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize Vosk model
    model_path = download_model()
    print(f"Using Vosk model at: {model_path}")
    
    # Set up command registry with all available commands
    command_registry = create_command_registry()
    
    # Initialize voice listener
    listener = VoiceListener(
        model_path=model_path,
        wake_word=WAKE_WORD,
        sample_rate=SAMPLE_RATE,
        buffer_size=BUFFER_SIZE
    )
    
    # Initialize command processor
    command_processor = CommandProcessor(command_registry)
    
    # Print available commands
    print("Available commands:")
    for command in command_registry.list_commands():
        param_info = ""
        if command.parameters:
            param_names = list(command.parameters.keys())
            param_info = f" [Parameters: {', '.join(param_names)}]"
        print(f"  - {command.name}: {command.description}{param_info}")
    
    print(f"\nVoice Command App - Say '{WAKE_WORD}' to activate, then speak a command")
    print("Press Ctrl+C to exit")
    print("-" * 50)
    assistant.speak("Hey! I'm you're helpful assistant. Looks like every things up and running."
                    "Just say, computer, any time you need me.", threaded=False)

    try:
        while True:
            # Listen for wake word
            if listener.listen_for_wake_word():
                assistant.speak("Ya")
                # Wake word detected, listen for command
                print("Listening for command...")
                audio_file = listener.record_audio(COMMAND_TIMEOUT)
                
                # Transcribe command with Vosk
                print("Transcribing...")
                transcription = listener.transcribe_audio(audio_file)
                
                if transcription:
                    print(f"Transcription: {transcription}")
                    
                    # Interpret command with OpenAI
                    print("Interpreting command...")
                    command_data = command_processor.interpret_command(transcription)
                    
                    if command_data["command"] != "unknown":
                        # Execute the command
                        command_registry.execute_command(
                            command_data["command"], 
                            command_data.get("parameters", {})
                        )
                    else:
                        assistant.speak("Sorry, I didn't understand that command.")
                else:
                    print("Failed to transcribe command.")

                listener.cleanup_audio_file(audio_file)
                
                print("\n" + "-"*50 + "\n")
    
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
