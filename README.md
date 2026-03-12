
# Simplificación Automática de Textos en Español con Modelos de Lenguaje

Proyecto de **Trabajo Terminal – Ingeniería en Inteligencia Artificial (ESCOM-IPN)**.

Este repositorio contiene el pipeline experimental para evaluar **modelos de lenguaje grandes (LLM)** en la tarea de **simplificación automática de textos en español**, con el objetivo de mejorar la **comprensión y legibilidad de documentos** mediante técnicas de procesamiento de lenguaje natural.

---

# Descripción del proyecto

La simplificación automática de textos busca transformar un texto complejo en una versión **más clara, comprensible y fácil de leer**, manteniendo su significado original.

Aplicaciones:

- accesibilidad para personas con dificultades lectoras
- comprensión de documentos técnicos
- educación
- lenguaje claro en instituciones públicas

Este proyecto forma parte del Trabajo Terminal:

**“Simplificación automática de textos en español para mejorar la comprensión y legibilidad de documentos mediante modelos de lenguaje”.**

---

# Objetivos

## Objetivo general

Desarrollar y evaluar un sistema de simplificación automática de textos en español utilizando modelos de lenguaje.

## Objetivos específicos

- Construir un pipeline experimental reproducible
- Evaluar distintos modelos de lenguaje
- Comparar técnicas de prompting
- Analizar métricas automáticas de simplificación
- Comparar resultados con trabajos previos

---

# Dataset

El proyecto utiliza el **dataset FEINA (Facilitating Easy-to-read Information in Spanish)**, un corpus paralelo diseñado para la tarea de **simplificación automática de textos en español**.

Este dataset contiene **pares alineados de oraciones**, donde cada registro incluye:

- **Texto original** (versión compleja)
- **Texto simplificado** (versión en lectura fácil)

El corpus fue creado para investigaciones en **accesibilidad lingüística y simplificación automática**, permitiendo entrenar y evaluar modelos de procesamiento de lenguaje natural orientados a generar textos más comprensibles.

---

## Uso en este proyecto

En este proyecto el dataset **FEINA** se utiliza para:

1. **Evaluar modelos de lenguaje grandes (LLM)** en tareas de simplificación textual.
2. Comparar las simplificaciones generadas por los modelos contra las versiones simplificadas del dataset.
3. Calcular métricas automáticas de simplificación como:

- **SARI**
- **BLEU**
- **ROUGE**
- **BERTScore**
- métricas de legibilidad (**Fernández-Huerta**, **INFLESZ**)

El dataset se utiliza principalmente en **modo evaluación**, sin realizar entrenamiento adicional de los modelos.


---

# Modelos evaluados

| Modelo | Tipo | Backend |
|------|------|------|
| Llama 3 | chat | Ollama |
| Mistral 7B | chat | Ollama |
| BLOOMZ | causal LM | Transformers |

Los modelos se evalúan **sin fine-tuning**, utilizando técnicas de **prompt engineering**.

---

# Estrategias de prompting

## Zero-shot
El modelo recibe solo instrucciones para simplificar.

## Few-shot
Se agregan ejemplos de simplificación para guiar al modelo.

## Simplificación guiada por reglas
Se agregan reglas lingüísticas como:

- usar vocabulario común
- evitar tecnicismos
- usar oraciones cortas
- preferir voz activa
- dividir oraciones largas

## Self-refine (experimental)
El modelo genera una simplificación, evalúa su resultado y produce una versión mejorada.

---

# Métricas de evaluación

## Replicación de trabajos previos

- SARI
- BLEU
- Fernández-Huerta
- Compression Ratio
- Sentence Splits
- Levenshtein Similarity
- Exact Copies
- Additions Proportion
- Deletions Proportion

## Métricas adicionales

- ROUGE
- INFLESZ
- BERTScore

## Evaluación semántica opcional

- Similaridad SBERT

---

# Arquitectura del pipeline

```bash
         Dataset
            ↓
      Prompt Builder
            ↓
LLM (Llama / Mistral / Bloom)
            ↓
    Postprocesamiento
            ↓
    Cálculo de métricas
            ↓
  Registro de experimentos
```

---

# Estructura del repositorio

```bash

TT2/
|
|  configs/
|  │
|  ├── models.py
|  │   Configuración de los modelos utilizados en los experimentos
|  │   (Mistral, Llama3, BLOOMZ) y sus respectivos backends
|  │   de inferencia (Ollama o HuggingFace Transformers).
|  │
|  ├── prompts.py
|  │   Definición de plantillas de prompts para simplificación:
|  │   - zero-shot
|  │   - few-shot
|  │
|  └── rules.py
|      Conjuntos de reglas de simplificación utilizadas en los
|      experimentos (R0–R4), que incluyen reglas léxicas,
|      sintácticas y semánticas.
|
|  data/
|  │
|  └── Dataset_FEINA.xlsx
|      Dataset utilizado en el proyecto para la evaluación de
|      simplificación de textos en español.
|
|  notebooks/
|  │
|  ├── 01_setup_experiments.ipynb
|  │   Configuración inicial del entorno experimental y pruebas
|  │   de inferencia con los modelos.
|  │
|  ├── 02_run_small_batch.ipynb
|  │   Ejecución de un lote pequeño de pruebas para validar
|  │   estabilidad del pipeline y calidad de generación.
|  │
|  ├── 03_run_medium_batch.ipynb
|  │   Ejecución de experimentos con un conjunto más grande
|  │   de textos para comparar modelos y configuraciones.
|  │
|  ├── 04_analyze_medium_batch.ipynb
|  │   Análisis de los resultados obtenidos en los experimentos
|  │   de tamaño medio.
|  │
|  └── 05_full_cv_experiments.ipynb
|      Ejecución completa de experimentos utilizando
|      validación cruzada sobre el dataset.
|
|  src/
|
|  inference/
|  │
|  ├── base.py
|  │   Interfaz base para generadores de modelos.
|  │
|  ├── factory.py
|  │   Factory que selecciona el generador adecuado según
|  │   el backend configurado.
|  │
|  ├── hf_generator.py
|  │   Implementación del generador usando
|  │   HuggingFace Transformers.
|  │
|  ├── ollama_generator.py
|  │   Implementación del generador usando Ollama
|  │   para modelos ejecutados localmente.
|  │
|  └── postprocess.py
|      Limpieza del texto generado por los modelos
|      para eliminar ruido o instrucciones filtradas.
|
|  experiment/
|  │
|  ├── runner.py
|  │   Núcleo del pipeline experimental. Se encarga de:
|  │   - construir prompts
|  │   - aplicar reglas de simplificación
|  │   - ejecutar inferencia en modelos
|  │   - medir tiempo de inferencia
|  │   - registrar resultados
|  │
|  └── schemas.py
|      Definición de estructuras de datos utilizadas
|      para registrar experimentos.
|
|  evaluation/
|  │
|  └── metrics.py
|      Implementación de métricas de evaluación de
|      simplificación textual, incluyendo métricas
|      de calidad, similitud y legibilidad.
|
|  utils/
|  │
|  ├── io.py
|  │   Funciones auxiliares para lectura y escritura
|  │   de archivos.
|  │
|  └── logging_utils.py
|      Sistema de registro de experimentos en formato
|      JSONL para garantizar reproducibilidad.
|
|  outputs/
|
|  ├── logs/
|  │   Bitácoras de experimentos ejecutados
|  │   (JSONL + manifest).
|  │
|  ├── metrics/
|  │   Resultados de evaluación de métricas
|  │   para cada experimento.
|  │
|  ├── generations/
|  │   Textos generados por los modelos.
|  │
|  └── cv_runs/
|      Resultados de validación cruzada.
|
|  README.md
|      Documentación principal del proyecto.
|
|  requirements.txt
|      Dependencias necesarias para ejecutar el proyecto.

```

---

# Notebooks del proyecto

Notebook 1 — Exploración del dataset

- inspección del corpus
- análisis de pares original/simplificado
- estadísticas básicas

Notebook 2 — Pruebas de modelos

- pruebas con Llama
- pruebas con Mistral
- pruebas con BLOOMZ
- comparación rápida

Notebook 3 — Pipeline experimental

- ejecución automática de experimentos
- generación de prompts
- ejecución de modelos
- registro de resultados

Notebook 4 — Evaluación de métricas

- cálculo de métricas automáticas
- SARI
- BLEU
- ROUGE
- INFLESZ
- BERTScore

Notebook 5 — Análisis de resultados

- comparación entre modelos
- análisis estadístico
- visualización de métricas
- selección del mejor enfoque

---

# Ejecución

Instalar dependencias:

```bash
pip install -r requirements.txt

Ejecutar experimentos:

python run_experiment.py
```
---

# Metodología

Se utiliza la metodología **CRISP-DM**:

1. Comprensión del problema
2. Comprensión de los datos
3. Preparación de los datos
4. Modelado
5. Evaluación
6. Despliegue

---

# Autores

- **Hariel Padilla Sánchez** - [HarielPS - @Github](https://github.com/HarielPS)
- **Nancy Aguilar Espinosa** - [Nancy523 - @Github](https://github.com/Nancy523)

Ingeniería en Inteligencia Artificial  
ESCOM – Instituto Politécnico Nacional

---

# Citation

@article{perez2023novel,
  title={A Novel Dataset for Financial Education Text Simplification in Spanish},
  author={Perez-Rojas, Nelson and Calderon-Ramirez, Saul and Solis-Salazar, Martin and Romero-Sandoval, Mario and Arias-Monge, Monica and Saggion, Horacio},
  journal={arXiv preprint arXiv:2312.09897},
  year={2023}
}

Paper: <https://arxiv.org/abs/2312.09897>

---
