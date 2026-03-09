from src.inference.ollama_generator import OllamaGenerator
from src.inference.hf_generator import HFGenerator


class GeneratorFactory:
    def __init__(self):
        self._ollama = None
        self._hf = None

    def get(self, backend: str):
        if backend == "ollama":
            if self._ollama is None:
                self._ollama = OllamaGenerator()
            return self._ollama

        if backend == "transformers":
            if self._hf is None:
                self._hf = HFGenerator()
            return self._hf

        raise ValueError(f"Backend no soportado: {backend}")

    def cleanup(self):
        if self._hf is not None:
            self._hf.unload_model()