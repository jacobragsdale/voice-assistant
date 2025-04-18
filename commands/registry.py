from typing import Dict, List, Any, Optional, Type
from commands.base_command import BaseCommand
from commands import available_commands

class CommandRegistry:
    def __init__(self):
        self._commands: Dict[str, BaseCommand] = {}
    
    def register(self, command: BaseCommand) -> None:
        """Register a command instance in the registry."""
        self._commands[command.name] = command
    
    def register_command_class(self, command_class: Type[BaseCommand]) -> None:
        """Instantiate and register a command class."""
        command_instance = command_class()
        self.register(command_instance)
    
    def get(self, name: str) -> Optional[BaseCommand]:
        """Get a command by name."""
        return self._commands.get(name)
    
    def list_commands(self) -> List[BaseCommand]:
        """Get a list of all registered commands."""
        return list(self._commands.values())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all commands for use with OpenAI."""
        return [cmd.to_dict() for cmd in self._commands.values()]
    
    def execute_command(self, command_name: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """Execute a command by name with the given parameters."""
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

def create_command_registry() -> CommandRegistry:
    """Create and return a CommandRegistry with all available commands."""
    registry = CommandRegistry()
    
    # Register all available commands
    for command_class in available_commands:
        registry.register_command_class(command_class)
    
    return registry 