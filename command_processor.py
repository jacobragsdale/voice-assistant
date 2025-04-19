import json
import os
from typing import Dict, List, Any, Set

import openai
from dotenv import load_dotenv

from commands.registry import CommandRegistry
from tts import VoiceAssistant

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class CommandProcessor:
    def __init__(self, command_registry: CommandRegistry):
        self.command_registry = command_registry

        # Load cached transcriptions
        self.transcription_cache_file = "transcription_cache.json"
        self.transcription_cache: Dict[str, Dict[str, Any]] = {}
        self._load_transcription_cache()
        self.assistant = VoiceAssistant()

    def interpret_command(self, transcription: str) -> Dict[str, Any]:
        # Check if this transcription was already interpreted and cached
        if transcription in self.transcription_cache:
            print(f"Cached command found. {self.transcription_cache[transcription]}")
            self.acknowledge_command(self.transcription_cache[transcription])
            return self.transcription_cache[transcription]

        # Get command schemas from registry
        command_schema = self.command_registry.get_schemas()

        # Get examples from each command
        examples = []
        for command in self.command_registry.list_commands():
            examples.extend(command.get_examples())

        # Limit to a reasonable number of examples (max 10-12)
        if len(examples) > 12:
            examples = self._balance_examples(examples)

        examples_formatted = "\n".join([
            f"User: \"{ex['transcription']}\"\nResponse: {json.dumps(ex['response'])}"
            for ex in examples
        ])

        try:
            # Create system prompt
            system_prompt = self._create_system_prompt(command_schema, examples_formatted)

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcription}
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
                temperature=0.1
            )

            # Extract and parse the JSON response
            result = json.loads(response.choices[0].message.content)

            # Check if the command is valid
            if "command" in result and self.command_registry.get(result["command"]):
                # Cache the transcription result for future use
                self.transcription_cache[transcription] = result
                self._save_transcription_cache()
                self.acknowledge_command(result)
                return result
            return {"command": "unknown"}

        except Exception as e:
            print(f"Error during command interpretation: {e}")
            return {"command": "unknown"}

    def acknowledge_command(self, command: Dict[str, Any]):
        print(command)
        if command["command"] == "unknown":
            self.assistant.speak("Sorry, I didn't understand that command.")
        elif command["command"] == "light" and command["parameters"].get("action") == "color":
            self.assistant.speak(f"Sure thing! Turning the lights {command['parameters'].get('color')} now.")

    def _balance_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance examples to have good command coverage"""
        balanced_examples: List[Dict[str, Any]] = []
        commands_covered: Set[str] = set()

        # First, ensure at least one example per command
        for example in examples:
            command_name = example["response"]["command"]
            if command_name not in commands_covered:
                balanced_examples.append(example)
                commands_covered.add(command_name)

            if len(balanced_examples) >= 6:
                break

        # Then add some examples with parameters
        for example in examples:
            if len(example["response"]["parameters"]) > 0 and len(balanced_examples) < 12:
                if example not in balanced_examples:
                    balanced_examples.append(example)

        return balanced_examples

    def _create_system_prompt(self, command_schema: List[Dict[str, Any]], examples_formatted: str) -> str:
        """Create the system prompt for OpenAI"""
        system_prompt = f"""
                You are a voice command interpreter that precisely identifies commands and extracts parameters from transcribed speech.
            
                AVAILABLE COMMANDS:
                {json.dumps(command_schema, indent=4)}
                
                YOUR TASK:
                1. Analyze the transcribed text to determine which command the user wants to execute
                2. Extract only the specific parameters defined for that command
                3. Handle potential speech recognition errors gracefully
                4. Return a clean JSON object with the command name and parameters
                
                IMPORTANT GUIDELINES:
                - If the user's intent matches one of the available commands, select that command
                - Extract only parameters that are defined for the selected command
                - If a parameter isn't mentioned, omit it from your response
                - If the command is unclear or doesn't match any available commands, respond with {{"command": "unknown"}}
                - Speech transcription may be imperfect; focus on the core intent rather than exact wording
                
                COLOR HANDLING:
                - When the user mentions changing the light color, set action to "color"
                - Capture the color description as precisely as possible in the "color" parameter
                - Examples of color descriptions: "blue", "warm white", "deep purple", "sunset orange"
                - Keep the color description simple but accurate to help with color conversion
                
                EXAMPLES:
                {examples_formatted}
                
                OUTPUT FORMAT:
                {{"command": "command_name", "parameters": {{"param1": "value1", "param2": "value2"}}}}"""
        return system_prompt

    def _load_transcription_cache(self) -> None:
        try:
            with open(self.transcription_cache_file, "r") as f:
                self.transcription_cache = json.load(f)
            print(f"Loaded {len(self.transcription_cache)} cached transcriptions.")
        except Exception as e:
            print(f"Error loading transcription cache: {e}")
            self.transcription_cache = {}

    def _save_transcription_cache(self) -> None:
        try:
            with open(self.transcription_cache_file, "w") as f:
                json.dump(self.transcription_cache, f, indent=4)
            print(f"Saved {len(self.transcription_cache)} cached transcriptions.")
        except Exception as e:
            print(f"Error saving transcription cache: {e}")