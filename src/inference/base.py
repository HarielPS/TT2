from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, model_name: str, generation_config: dict) -> str:
        pass