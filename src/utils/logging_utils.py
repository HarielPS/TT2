from pathlib import Path
from src.utils.io import append_jsonl, write_json, ensure_dir


class ExperimentLogger:
    def __init__(self, base_dir: str = "outputs/logs"):
        self.base_dir = Path(base_dir)
        ensure_dir(self.base_dir)

    def save_manifest(self, experiment_id: str, manifest: dict):
        path = self.base_dir / f"{experiment_id}_manifest.json"
        write_json(path, manifest)

    def append_record(self, experiment_id: str, record: dict):
        path = self.base_dir / f"{experiment_id}.jsonl"
        append_jsonl(path, record)