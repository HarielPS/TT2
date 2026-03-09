from dataclasses import dataclass, asdict, field
from typing import Optional
from datetime import datetime
import uuid


@dataclass
class ExperimentRecord:
    experiment_id: str
    run_id: str
    timestamp: str

    dataset_name: str
    fold_id: Optional[int]
    split_name: Optional[str]

    model_key: str
    model_id: str
    backend: str              # ollama | transformers
    prompt_type: str          # zero-shot | few-shot
    ruleset: str              # R0 ... R4

    few_shot_example_ids: list = field(default_factory=list)
    generation_config: dict = field(default_factory=dict)

    sample_id: Optional[str] = None
    source_text: str = ""
    reference_text: Optional[str] = None
    generated_text: Optional[str] = None

    prompt_text: Optional[str] = None
    inference_seconds: Optional[float] = None
    status: str = "created"
    error_message: Optional[str] = None

    metrics: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def new_record(
    experiment_id: str,
    dataset_name: str,
    model_key: str,
    model_id: str,
    backend: str,
    prompt_type: str,
    ruleset: str,
    generation_config: dict,
    source_text: str,
    reference_text: str | None = None,
    sample_id: str | None = None,
    fold_id: int | None = None,
    split_name: str | None = None,
    few_shot_example_ids: list | None = None,
    prompt_text: str | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=experiment_id,
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        dataset_name=dataset_name,
        fold_id=fold_id,
        split_name=split_name,
        model_key=model_key,
        model_id=model_id,
        backend=backend,
        prompt_type=prompt_type,
        ruleset=ruleset,
        few_shot_example_ids=few_shot_example_ids or [],
        generation_config=generation_config,
        sample_id=sample_id,
        source_text=source_text,
        reference_text=reference_text,
        prompt_text=prompt_text,
    )