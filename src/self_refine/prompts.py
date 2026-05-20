# src/self_refine/prompts.py

SELF_REFINE_PROMPT_TEMPLATE = """
Eres un editor de textos en español especializado en simplificación de textos formales.

Tu tarea es mejorar una simplificación ya generada. No debes rehacerla desde cero.

Texto original:
{source_text}

Simplificación actual:
{generated_text}

Instrucciones obligatorias:
1. Responde únicamente en español.
2. Devuelve solo el texto refinado.
3. No escribas notas, explicaciones, comentarios ni justificaciones.
4. No uses frases como "Note:", "Nota:", "He cambiado", "No hice cambios", "Ningún cambio significativo" o similares.
5. Conserva el significado del texto original.
6. No inventes información.
7. No elimines datos importantes.
8. Si la simplificación actual ya es clara y fiel, devuelve una versión limpia muy parecida, sin explicar nada.
9. Si la simplificación actual es casi igual al texto original, reescríbela con palabras más simples.
10. Sustituye palabras formales o técnicas por palabras más comunes cuando sea posible.
11. Usa oraciones cortas.
12. Usa viñetas solo si el texto contiene listas, pasos, causas, consecuencias o varias ideas separadas.
13. Conserva números, fechas, nombres propios y cantidades.
14. No agregues información que no esté en el texto original.

Texto refinado:
""".strip()


def build_self_refine_prompt(source_text: str, generated_text: str) -> str:
    return SELF_REFINE_PROMPT_TEMPLATE.format(
        source_text=source_text,
        generated_text=generated_text,
    )