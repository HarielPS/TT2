import time
import traceback
import gc
import torch

from configs.models import MODELS
from configs.prompts import ZERO_SHOT_TEMPLATE, build_few_shot_prompt
from configs.rules import RULESETS
from src.experiment.schemas import new_record
from src.inference.factory import GeneratorFactory
from src.inference.postprocess import clean_generated_text
from src.utils.logging_utils import ExperimentLogger


class ExperimentRunner:
    def __init__(self, experiment_id: str, log_dir: str = "outputs/logs"):
        self.experiment_id = experiment_id
        self.logger = ExperimentLogger(base_dir=log_dir)
        self.factory = GeneratorFactory()

    def build_prompt(
        self,
        source_text: str,
        prompt_type: str,
        ruleset: str,
        few_shot_examples=None,
    ) -> str:
        rules = RULESETS.get(ruleset, [])
        rules_block = ""

        if rules:
            compact_rules = "; ".join(rules)
            rules_block = f"Guía interna de simplificación: {compact_rules}.\n"

        if prompt_type == "zero-shot":
            prompt = ZERO_SHOT_TEMPLATE.format(
                source=source_text,
                rules_block=rules_block,
            )
        elif prompt_type == "few-shot":
            prompt = build_few_shot_prompt(
                source=source_text,
                examples=few_shot_examples or [],
                rules_block=rules_block,
            )
        else:
            raise ValueError(f"prompt_type no soportado: {prompt_type}")

        return prompt

    def run_one(
        self,
        dataset_name: str,
        model_key: str,
        prompt_type: str,
        ruleset: str,
        source_text: str,
        reference_text: str = None,
        sample_id: str = None,
        fold_id: int = None,
        split_name: str = None,
        few_shot_examples: list = None,
        few_shot_example_ids: list = None,
        generation_config: dict = None,
    ):
        model_cfg = MODELS[model_key]
        backend = model_cfg["backend"]
        runtime_name = model_cfg["runtime_name"]
        generation_config = generation_config or {}

        prompt = self.build_prompt(
            source_text=source_text,
            prompt_type=prompt_type,
            ruleset=ruleset,
            few_shot_examples=few_shot_examples,
        )

        record = new_record(
            experiment_id=self.experiment_id,
            dataset_name=dataset_name,
            model_key=model_key,
            model_id=model_cfg["model_id"],
            backend=backend,
            prompt_type=prompt_type,
            ruleset=ruleset,
            generation_config=generation_config,
            source_text=source_text,
            reference_text=reference_text,
            sample_id=sample_id,
            fold_id=fold_id,
            split_name=split_name,
            few_shot_example_ids=few_shot_example_ids or [],
            prompt_text=prompt,
        )

        try:
            generator = self.factory.get(backend)
            start = time.perf_counter()

            if backend == "ollama":
                raw_generated = generator.generate(
                    prompt=prompt,
                    model_name=runtime_name,
                    generation_config=generation_config,
                    system_prompt=(
                        "Simplifica textos en español. "
                        "Responde solo con la versión simplificada final. "
                        "No repitas instrucciones ni reglas."
                    ),
                )
            else:
                raw_generated = generator.generate(
                    prompt=prompt,
                    model_name=runtime_name,
                    generation_config=generation_config,
                )

            generated = clean_generated_text(raw_generated)
            elapsed = time.perf_counter() - start

            record.generated_text = generated
            record.inference_seconds = round(elapsed, 4)
            record.status = "success"

        except Exception as e:
            record.status = "error"
            record.error_message = (
                f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )

        self.logger.append_record(self.experiment_id, record.to_dict())

        self._light_cleanup()
        return record

    def _light_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def full_cleanup(self):
        self.factory.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()