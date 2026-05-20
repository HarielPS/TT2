# src/self_refine/refiner.py

import time
import requests

from configs.models import MODELS, DEFAULT_GENERATION_CONFIG
from src.self_refine.prompts import build_self_refine_prompt
from src.self_refine.schemas import SelfRefineResult


class OllamaSelfRefiner:
    def __init__(
        self,
        model_key: str = "llama3",
        host: str = "http://localhost:11434",
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 120,
    ):
        if model_key not in MODELS:
            raise ValueError(
                f"Modelo no encontrado en configs.models.MODELS: {model_key}. "
                f"Disponibles: {list(MODELS.keys())}"
            )

        model_cfg = MODELS[model_key]

        if model_cfg.get("backend") != "ollama":
            raise ValueError(
                f"Self-Refine requiere backend='ollama'. "
                f"El modelo {model_key} tiene backend={model_cfg.get('backend')}"
            )

        self.model_key = model_key
        self.model_name = model_cfg["runtime_name"]
        self.model_id = model_cfg["model_id"]
        self.host = host.rstrip("/")

        self.temperature = (
            DEFAULT_GENERATION_CONFIG.get("temperature", 0.2)
            if temperature is None
            else temperature
        )
        self.top_p = (
            DEFAULT_GENERATION_CONFIG.get("top_p", 0.9)
            if top_p is None
            else top_p
        )
        self.max_tokens = (
            DEFAULT_GENERATION_CONFIG.get("max_new_tokens", 256)
            if max_tokens is None
            else max_tokens
        )
        self.timeout = timeout

    def refine(self, source_text: str, generated_text: str) -> SelfRefineResult:
        prompt_text = build_self_refine_prompt(
            source_text=source_text,
            generated_text=generated_text,
        )

        start = time.time()

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt_text,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=self.timeout,
            )

            response.raise_for_status()
            data = response.json()

            refined_text = data.get("response", "").strip()
            refined_text = self._clean_output(refined_text)

            return SelfRefineResult(
                source_text=source_text,
                generated_text=generated_text,
                refined_text=refined_text,
                prompt_text=prompt_text,
                model_name=self.model_name,
                status="ok",
                error_message=None,
                inference_seconds=time.time() - start,
            )

        except Exception as e:
            return SelfRefineResult(
                source_text=source_text,
                generated_text=generated_text,
                refined_text="",
                prompt_text=prompt_text,
                model_name=self.model_name,
                status="error",
                error_message=str(e),
                inference_seconds=time.time() - start,
            )

    def _clean_output(self, text: str) -> str:
        """
        Limpia salidas donde el modelo agrega explicaciones no deseadas.
        """
        if not text:
            return text

        text = text.strip()

        bad_prefixes = [
            "Versión refinada:",
            "Texto refinado:",
            "Simplificación refinada:",
            "Resultado:",
            "Aquí tienes la versión refinada:",
            "Aquí está la versión refinada:",
        ]

        for prefix in bad_prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        stop_markers = [
            "\n\nNote:",
            "\nNote:",
            "\n\n(Note:",
            "\n(Note:",
            "\n\nNota:",
            "\nNota:",
            "\n\n(Nota:",
            "\n(Nota:",
            "\n\nComentario:",
            "\nComentario:",
            "\n\nExplicación:",
            "\nExplicación:",
            "\n\nHe ",
            "\nHe ",
            "\n\nNo hice",
            "\nNo hice",
            "\n\nNingún cambio",
            "\nNingún cambio",
            "\n\n(",
        ]

        for marker in stop_markers:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx].strip()

        if text.lower().startswith("(note:") or text.lower().startswith("(nota:"):
            return ""

        return text.strip()