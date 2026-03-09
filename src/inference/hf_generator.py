import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.inference.base import BaseGenerator


class HFGenerator(BaseGenerator):
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.current_model_id = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_id: str):
        if self.current_model_id == model_id and self.model is not None:
            return

        self.unload_model()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        self.model = self.model.to(self.device)
        self.model.eval()
        self.current_model_id = model_id

    def generate(self, prompt: str, model_name: str, generation_config: dict) -> str:
        self.load_model(model_name)

        prompt = "" if prompt is None else str(prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            return_attention_mask=True,
            add_special_tokens=True,
        )

        if inputs["input_ids"].shape[1] == 0:
            inputs = {
                "input_ids": torch.tensor([[self.tokenizer.eos_token_id]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
            }

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        do_sample = generation_config.get("do_sample", False)

        gen_kwargs = {
            "max_new_tokens": generation_config.get("max_new_tokens", 128),
            "repetition_penalty": generation_config.get("repetition_penalty", 1.05),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": False,
        }

        if do_sample:
            gen_kwargs["temperature"] = generation_config.get("temperature", 0.2)
            gen_kwargs["top_p"] = generation_config.get("top_p", 0.9)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.current_model_id = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()