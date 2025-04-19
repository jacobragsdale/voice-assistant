from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


class BaseCommand(ABC):
    def __init__(self, name: str, description: str, parameters: Dict[str, Dict[str, str]],
                 example_queries: Optional[List[Dict[str, Any]]] = None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.example_queries = example_queries or self._generate_default_examples()

    def _generate_default_examples(self) -> List[Dict[str, Any]]:
        examples = [{"query": f"{self.name}", "parameters": {}}]

        for param_name, param_info in self.parameters.items():
            examples.append({
                "query": f"{self.name} with {param_name} {param_info.get('default', 'example')}",
                "parameters": {param_name: param_info.get('default', 'example')}
            })

        return examples

    @abstractmethod
    def execute(self, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the command with the given parameters."""
        pass

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
