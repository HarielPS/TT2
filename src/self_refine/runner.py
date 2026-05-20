# src/self_refine/runner.py

from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

from src.self_refine.refiner import OllamaSelfRefiner


def run_self_refine_dataframe(
    df: pd.DataFrame,
    source_col: str = "source_text",
    generated_col: str = "generated_text",
    system_col: str = "system_name",
    id_cols: Optional[list[str]] = None,
    refiner_model_key: str = "llama3",
    output_system_suffix: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Aplica Self-Refine a un DataFrame de simplificaciones.

    Espera columnas:
    - source_text
    - generated_text
    - system_name

    Devuelve:
    - generated_text_base
    - generated_text_refined
    - refined_system_name
    - status
    - error_message
    - inference_seconds
    """

    if id_cols is None:
        id_cols = []

    if output_system_suffix is None:
        output_system_suffix = f"_self_refine_{refiner_model_key}"

    required_cols = [source_col, generated_col, system_col] + id_cols
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    refiner = OllamaSelfRefiner(
        model_key=refiner_model_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Self-Refine"):
        source_text = row[source_col]
        generated_text = row[generated_col]

        result = refiner.refine(
            source_text=source_text,
            generated_text=generated_text,
        )

        output_row = {}

        for col in id_cols:
            output_row[col] = row[col]

        output_row["base_system_name"] = row[system_col]
        output_row["refined_system_name"] = f"{row[system_col]}{output_system_suffix}"

        output_row["source_text"] = source_text

        if "reference_text" in df.columns:
            output_row["reference_text"] = row["reference_text"]

        output_row["generated_text_base"] = generated_text
        output_row["generated_text_refined"] = result.refined_text

        output_row["self_refine_model_key"] = refiner_model_key
        output_row["self_refine_model_runtime"] = result.model_name
        output_row["self_refine_prompt"] = result.prompt_text
        output_row["status"] = result.status
        output_row["error_message"] = result.error_message
        output_row["inference_seconds"] = result.inference_seconds

        rows.append(output_row)

    out = pd.DataFrame(rows)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_path, index=False, encoding="utf-8-sig")

    return out