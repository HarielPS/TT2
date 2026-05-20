# src/self_refine/schemas.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class SelfRefineResult:
    source_text: str
    generated_text: str
    refined_text: str
    prompt_text: str
    model_name: str
    status: str
    error_message: Optional[str] = None
    inference_seconds: Optional[float] = None