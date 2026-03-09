# SIMPLIFICACIÓN AUTOMÁTICA DE TEXTOS EN ESPAÑOL PARA MEJORAR LA COMPRENSIÓN Y LEGIBILIDAD DE DOCUMENTOS MEDIANTE MODELOS DE LENGUAJE

## Dataset

This project uses the **FEINA v1.0** dataset for evaluating the performance of automatic **Spanish financial text simplification models**.

FEINA is a dataset composed of approximately **5,312 sentence pairs**, where each pair contains:

* a **complex financial sentence**
* a **simplified version** of the same sentence.

The dataset was created as part of a research project at the **Instituto Tecnológico de Costa Rica**, in collaboration with **Universitat Pompeu Fabra**, and is publicly available through Hugging Face.

Dataset repository:
<https://huggingface.co/datasets/saul1917/FEINA>

In this repository, the dataset is used **only for experimentation and evaluation of text simplification models**.

All credit for the dataset creation belongs to the original authors.

---

## Resultados

Debe guardar también:

fecha/hora

nombre del experimento

dataset

fold

split

modelo

backend (ollama o transformers)

prompt type (zero-shot / few-shot)

rules level (R0 a R4)

ids de ejemplos few-shot

hiperparámetros de generación

texto original

referencia

salida generada

tiempo de inferencia

luego métricas

Yo te recomiendo usar:

JSONL para bitácora corrida por corrida

CSV/Parquet para resumen tabular

---

## Citation

@article{perez2023novel,
  title={A Novel Dataset for Financial Education Text Simplification in Spanish},
  author={Perez-Rojas, Nelson and Calderon-Ramirez, Saul and Solis-Salazar, Martin and Romero-Sandoval, Mario and Arias-Monge, Monica and Saggion, Horacio},
  journal={arXiv preprint arXiv:2312.09897},
  year={2023}
}

Paper: <https://arxiv.org/abs/2312.09897>

---

## Comandos

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -U pip
   pip install requests transformers accelerate bitsandbytes sentencepiece
