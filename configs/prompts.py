ZERO_SHOT_TEMPLATE = """
Reescribe en español el siguiente texto con lenguaje más claro y sencillo.
Conserva el significado original y no inventes información.
{rules_block}
Devuelve solo la versión final simplificada.

Texto:
{source}

Versión simplificada:
""".strip()


FEW_SHOT_PREFIX = """
Reescribe en español cada texto con lenguaje más claro y sencillo.
Conserva el significado original y no inventes información.
{rules_block}
Devuelve solo la versión final simplificada.
""".strip()


def build_few_shot_prompt(source: str, examples: list[dict], rules_block: str = "") -> str:
    prefix = FEW_SHOT_PREFIX.format(rules_block=rules_block)

    blocks = [prefix]

    for i, ex in enumerate(examples, start=1):
        blocks.append(
            f"\nEjemplo {i}\n"
            f"Texto: {ex['source']}\n"
            f"Versión simplificada: {ex['target']}"
        )

    blocks.append(
        f"\nTexto: {source}\n"
        f"Versión simplificada:"
    )

    return "\n".join(blocks)