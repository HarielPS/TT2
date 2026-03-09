import requests
from typing import Optional
from src.inference.base import BaseGenerator


class OllamaGenerator(BaseGenerator):
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        model_name: str,
        generation_config: dict,
        system_prompt: Optional[str] = None,
    ) -> str:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0,
            "options": {
                "temperature": generation_config.get("temperature", 0.2),
                "top_p": generation_config.get("top_p", 0.9),
                "num_predict": generation_config.get("max_new_tokens", 256),
                "repeat_penalty": generation_config.get("repetition_penalty", 1.05),
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()