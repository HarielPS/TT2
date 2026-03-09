import re


def clean_generated_text(text: str) -> str:
    if text is None:
        return None

    t = text.strip()

    # Quitar prefijos comunes al inicio
    prefixes = [
        "texto simplificado:",
        "texto simplificado final:",
        "versión simplificada:",
        "simplificación:",
        "respuesta:",
        "salida:",
    ]

    changed = True
    while changed:
        changed = False
        lower_t = t.lower()
        for p in prefixes:
            if lower_t.startswith(p):
                t = t[len(p):].strip()
                changed = True
                break

    # Cortar desde patrones típicos de fuga del prompt o notas añadidas
    cut_patterns = [
        r"\btexto simplificado\b.*",
        r"\btexto simplificado final\b.*",
        r"\bversión simplificada\b.*",
        r"\bguía interna de simplificación\b.*",
        r"\busa palabras comunes\b.*",
        r"\bevita tecnicismos\b.*",
        r"\breemplaza palabras difíciles\b.*",
        r"\busa oraciones cortas\b.*",
        r"\bdivide oraciones largas\b.*",
        r"\bno menciones\b.*",
        r"\bnote\s*:.*",          # Note:
        r"\bnota\s*:.*",          # Nota:
        r"\bexplicación\s*:.*",   # Explicación:
        r"\bexplanation\s*:.*",   # Explanation:
    ]

    for pat in cut_patterns:
        m = re.search(pat, t, flags=re.IGNORECASE | re.DOTALL)
        if m:
            t = t[:m.start()].strip()

    # También elimina notas entre paréntesis al final:
    # (Note: ...)
    t = re.sub(
        r"\(\s*(note|nota|explanation|explicación)\s*:.*?\)\s*$",
        "",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Compactar espacios
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)

    return t.strip()