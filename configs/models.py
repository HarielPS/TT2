MODELS = {
    "bloomz_small": {
        "model_id": "bigscience/bloomz-560m",
        "kind": "plain",
        "backend": "transformers",
        "runtime_name": "bigscience/bloomz-560m",
    },

    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "kind": "chat",
        "backend": "ollama",
        "runtime_name": "mistral:7b-instruct-v0.2-q4_0",
    },

    "llama3": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "kind": "chat",
        "backend": "ollama",
        "runtime_name": "llama3:8b",
    },
    # "phi3": {
    #     "model_id": "microsoft/Phi-3-mini-4k-instruct",
    #     "kind": "chat",
    #     "backend": "ollama",
    #     "runtime_name": "phi3",
    # },
}
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_new_tokens": 256,
    "repetition_penalty": 1.05,
    "do_sample": False,
}