import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "bigscience/bloomz-560m"

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)

tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

model = model.to("cuda")
model.eval()

prompt = "Hola, explícame qué es una red neuronal."
inputs = tok(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=256,
    return_attention_mask=True,
)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.inference_mode():
    outputs = model(**inputs)
    print("logits shape:", outputs.logits.shape)

    out = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        use_cache=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

print(tok.decode(out[0], skip_special_tokens=True))