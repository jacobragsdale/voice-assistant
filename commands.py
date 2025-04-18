import time
from typing import Dict, List, Any, Callable, Optional, Union

class Command:
    """A class representing a voice command with parameters."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Dict[str, str]], 
                 action: Callable[[Dict[str, Any]], Any], example_queries: Optional[List[Dict[str, Any]]] = None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.action = action
        self.example_queries = example_queries or self._generate_default_examples()
    
    def _generate_default_examples(self) -> List[Dict[str, Any]]:
        examples = []
        
        # Add a basic example with no parameters
        examples.append({
            "query": f"{self.name}",
            "parameters": {}
        })
        
        # Add examples for each parameter
        for param_name, param_info in self.parameters.items():
            examples.append({
                "query": f"{self.name} with {param_name} {param_info.get('default', 'example')}",
                "parameters": {param_name: param_info.get('default', 'example')}
            })
        
        return examples
    
    def execute(self, parameters: Optional[Dict[str, Any]] = None) -> Any:
        if parameters is None:
            parameters = {}
        return self.action(parameters)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def get_examples(self) -> List[Dict[str, Any]]:
        return [
            {
                "transcription": example["query"],
                "response": {"command": self.name, "parameters": example["parameters"]}
            }
            for example in self.example_queries
        ]

class CommandRegistry:
    """A registry for managing available commands."""
    
    def __init__(self):
        """Initialize a new command registry."""
        self._commands: Dict[str, Command] = {}
    
    def register(self, command: Command) -> None:
        self._commands[command.name] = command
    
    def get(self, name: str) -> Optional[Command]:
        return self._commands.get(name)
    
    def list_commands(self) -> List[Command]:
        return list(self._commands.values())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        return [cmd.to_dict() for cmd in self._commands.values()]
    
    def execute_command(self, command_name: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        command = self.get(command_name)
        if command:
            print(f"Executing command: {command_name}")
            if parameters:
                param_str = ", ".join(f"{k}='{v}'" for k, v in parameters.items())
                print(f"With parameters: {param_str}")
            
            command.execute(parameters)
            return True
        else:
            print(f"Unknown command: {command_name}")
            return False

# Create default commands
def create_default_commands() -> CommandRegistry:
    """Create and return a CommandRegistry with default commands."""
    registry = CommandRegistry()
    
    # Weather command
    registry.register(Command(
        name="weather",
        description="Check the current weather for a specific location",
        parameters={
            "location": {"type": "string", "description": "The city or location to get weather for", "default": "current location"}
        },
        action=lambda params: print(f"Getting weather information for {params.get('location', 'your area')}: It's currently 72°F and sunny"),
        example_queries=[
            {"query": "what's the weather like in San Francisco", "parameters": {"location": "San Francisco"}},
            {"query": "check weather", "parameters": {}},
            {"query": "how's the weather today", "parameters": {}},
            {"query": "weather forecast for Chicago", "parameters": {"location": "Chicago"}}
        ]
    ))
    
    # Time command
    registry.register(Command(
        name="time",
        description="Tell the current time, optionally for a specific timezone",
        parameters={
            "timezone": {"type": "string", "description": "The timezone to check the time for", "default": "local"}
        },
        action=lambda params: print(f"The current time is {time.strftime('%I:%M %p')} {params.get('timezone', 'local time')}"),
        example_queries=[
            {"query": "what time is it in Tokyo", "parameters": {"timezone": "Tokyo"}},
            {"query": "tell me the current time", "parameters": {}},
            {"query": "what's the time in London", "parameters": {"timezone": "London"}},
            {"query": "check time", "parameters": {}}
        ]
    ))
    
    # Joke command
    registry.register(Command(
        name="joke",
        description="Tell a joke, optionally about a specific topic",
        parameters={
            "topic": {"type": "string", "description": "The topic of the joke", "default": "general"}
        },
        action=lambda params: print(f"Here's a {params.get('topic', 'random')} joke: Why don't scientists trust atoms? Because they make up everything!"),
        example_queries=[
            {"query": "tell me a joke about computers", "parameters": {"topic": "computers"}},
            {"query": "i want to hear a joke", "parameters": {}},
            {"query": "joke about programming", "parameters": {"topic": "programming"}},
            {"query": "say something funny", "parameters": {}}
        ]
    ))
    
    return registry 