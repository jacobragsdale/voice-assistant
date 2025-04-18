# Voice Command App

A Python application that listens for a wake word ("computer"), then processes voice commands using local transcription with Vosk and command interpretation with OpenAI.

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key in the `.env` file (required for command interpretation):

```
OPENAI_API_KEY=your-api-key
```

3. Download the Vosk model (automatic on first run):

```bash
python download_model.py
```

Note: Installing dependencies can sometimes be challenging:
- On macOS, you may need to install portaudio first: `brew install portaudio`
- On Windows, you might need wheel files for PyAudio
- On Linux, try: `sudo apt-get install python3-pyaudio`

## Usage

### Testing Microphone Setup

Before running the main app, you can test your microphone and wake word detection:

```bash
python test_microphone.py
```

This will listen for the wake word "computer" and confirm if it was detected.

### Running the Full App

Run the voice command app:

```bash
python voice_command_app.py
```

How it works:
1. The app listens continuously for the wake word "computer"
2. When detected, it will record audio for 3 seconds to capture your command
3. The command is transcribed locally using Vosk
4. OpenAI is used to interpret which command you want to execute and extract parameters
5. The app executes the corresponding command with the provided parameters

Press Ctrl+C to exit the application.

## Available Commands

The app comes with the following built-in commands:
- **weather [location]**: Check the current weather for a specific location
- **time [timezone]**: Tell the current time, optionally for a specific timezone
- **joke [topic]**: Tell a joke, optionally about a specific topic

Examples of voice commands:
- "What's the weather in New York?"
- "Tell me the time in Tokyo"
- "Tell me a joke about programming"

## Project Structure

The application has been modularized into the following components:

- **voice_command_app.py**: Main application entry point
- **listener.py**: Handles audio recording, wake word detection, and transcription
- **commands.py**: Defines the Command class and CommandRegistry
- **command_processor.py**: Uses OpenAI to interpret commands and extract parameters
- **download_model.py**: Downloads the Vosk speech recognition model
- **test_microphone.py**: Tool to test microphone input and wake word detection

## Adding New Commands

To add new commands, edit the `create_default_commands` function in `commands.py`:

```python
def create_default_commands():
    registry = CommandRegistry()
    
    # Existing commands...
    
    # Add your new command
    registry.register(Command(
        name="my_command",
        description="Description of what your command does",
        parameters={
            "param1": {"type": "string", "description": "Description of this parameter", "default": "default value"},
            "param2": {"type": "string", "description": "Another parameter description", "default": "default value"}
        },
        action=lambda params: print(f"Executing with {params.get('param1')} and {params.get('param2')}")
    ))
    
    return registry
```

## Technology

This app uses:
- **Vosk**: For local speech-to-text (wake word detection and command transcription)
- **OpenAI**: For natural language understanding (interpreting commands and extracting parameters)
- **PyAudio**: For recording audio from your microphone 