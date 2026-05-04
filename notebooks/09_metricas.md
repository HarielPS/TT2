# README - como se manejaron las metricas en el proyecto

En esta parte del proyecto no se quiso evaluar los resultados con una sola metrica porque simplificar un texto no depende de una sola cosa. Un texto puede parecer muy parecido a la referencia pero perder significado. O puede ser muy facil de leer pero cambiar demasiado el contenido. Por eso se decidio usar varias metricas y luego separar cuales eran las principales y cuales quedaban como apoyo.

## idea general

La evaluacion se penso en dos niveles.

El primero fue mantener comparacion con FEINA y con lo que ya reporta el paper. El segundo fue agregar metricas extra para que el analisis del trabajo fuera mas completo y no quedarse solo con las metricas originales del paper. En el reporte tambien se dejo claro que se tomarian metricas del estudio original y se complementarian con otras metricas comunes en simplificacion automatica de texto.

## por que no se uso una sola metrica

No se uso una sola metrica porque cada una mira algo distinto.

- unas miden que tanto se simplifico el texto
- otras miden que tanto se parece a la referencia humana
- otras miden legibilidad
- otras miden si se conservo el significado
- otras ayudan a ver si el modelo solo copio o si realmente hizo cambios

Por eso el script de evaluacion termina calculando varias metricas y no solo dos. En `metrics.py` el pipeline resume 19 metricas o campos metricos. :contentReference[oaicite:2]{index=2}

## metricas que vienen de FEINA o que se tomaron para replica

Para poder compararnos con FEINA se retomaron metricas que ya aparecen en el paper o en el bloque de replicacion del trabajo.

Las principales de comparacion con FEINA fueron

- SARI
- BLEU
- Fernandez Huerta

En el mismo resumen del trabajo tambien se incluyeron otras metricas del bloque FEINA como

- compression ratio
- sentence splits
- levenshtein similarity
- exact copies
- additions proportion
- deletions proportion

Estas metricas sirven para ver cosas distintas.

### SARI

Se uso porque es una de las metricas mas importantes en simplificacion automatica. Sirve para evaluar si el sistema agrega elimina o conserva palabras de forma parecida a una simplificacion humana. En tu propio resumen se menciona que FEINA reporta SARI como metrica central de simplificacion. Por eso se mantuvo como una de las mas importantes del proyecto. :contentReference[oaicite:3]{index=3}

### BLEU

Se uso porque permite medir similitud con la referencia humana. No es suficiente por si sola para simplificacion pero si ayuda a tener una comparacion directa con FEINA y con otros resultados reportados. Por eso se dejo como metrica de comparacion con el paper, aunque no fue la principal para seleccionar configuraciones.

### Fernandez Huerta

Se uso porque es una metrica de legibilidad para español. Ayuda a ver si el texto simplificado se vuelve mas facil de leer. Se mantuvo porque en FEINA tambien se reporta y porque en este trabajo interesaba no solo comparar con la referencia sino revisar si realmente mejoraba la lectura del texto final.

### Compression ratio

Se uso para ver cuanto se reduce el texto con respecto al original. Sirve para detectar si el modelo no esta simplificando casi nada o si por el contrario esta recortando demasiado.

### Sentence splits

Se uso para medir si el modelo divide oraciones complejas en otras mas simples. Eso es importante porque una parte de la simplificacion no solo es cambiar palabras sino tambien reorganizar la estructura de las oraciones.

### Levenshtein similarity

Se uso para medir cuanto cambia el texto respecto al original. Ayuda a ver si el modelo casi no lo toca o si lo modifica demasiado.

### Exact copy

Se uso para detectar cuando el modelo devuelve practicamente el mismo texto original sin simplificar. Esta metrica fue util porque despues incluso se filtro que las configuraciones con demasiada copia exacta no se consideraran buenas opciones.

### Additions proportion y Deletions proportion

Se usaron para ver cuanto agrega y cuanto elimina el modelo. Esto ayuda a entender el comportamiento del sistema y no solo quedarse con un puntaje final.

## metricas que agregamos nosotros como extension

Ademas de las metricas del paper se propusieron otras metricas para enriquecer la evaluacion del trabajo. En el resumen del reporte aparece claro que como extension del trabajo se agregaron

- ROUGE
- INFLESZ
- BERTScore

y como opcional se menciono SBERT o evaluacion humana. :contentReference[oaicite:11]{index=11}

### ROUGE

Se metio porque ayuda a medir que tanto se parece el texto generado a la referencia humana en terminos de palabras y estructura. No se uso como metrica principal para elegir configuraciones pero si fue util como apoyo para ver si la salida se acercaba a la forma en que un humano simplifico el texto.

En el script ROUGE no aparece como una sola columna sino dividido en tres variantes

- `rouge1_f`
- `rouge2_f`
- `rougeL_f`

Esto pasa porque ROUGE no es una sola medida unica sino una familia de metricas.

#### `rouge1_f`

Mide el traslape de palabras individuales entre la prediccion y la referencia. Sirve para ver si ambas comparten vocabulario parecido de forma general.

#### `rouge2_f`

Mide el traslape de pares de palabras consecutivas. Es un poco mas estricta porque no solo revisa si aparecen las mismas palabras sino si aparecen juntas en fragmentos cortos parecidos.

#### `rougeL_f`

Mide el parecido tomando en cuenta la subsecuencia comun mas larga. En terminos simples ayuda a ver si la salida conserva una estructura parecida a la referencia y no solo palabras sueltas.

### INFLESZ

Se metio porque es una metrica de legibilidad pensada para español y servia para revisar si el texto simplificado realmente quedaba mas facil de leer que el original. A diferencia de otras metricas que comparan contra una referencia humana, INFLESZ ayuda a observar directamente que tan accesible se ve el texto resultante por su forma de escribir.

En el script no se guarda como una sola columna sino en tres

- `inflesz_pred`
- `inflesz_src`
- `inflesz_delta`

Esto se hizo para no quedarnos solo con el valor del texto simplificado sino poder compararlo contra el texto original.

#### `inflesz_pred`

    Es el valor de INFLESZ del texto generado por el modelo. Sirve para ver que tan legible quedo la salida final.

#### `inflesz_src`

    Es el valor de INFLESZ del texto original. Sirve como punto de referencia para saber desde donde se estaba partiendo.

#### `inflesz_delta`

    Es la diferencia entre ambos valores. Esta parte es importante porque permite ver si de verdad hubo mejora en legibilidad y no solo conocer el puntaje aislado del texto simplificado.

Por eso cuando en el reporte se dice INFLESZ en realidad en el codigo se guardo de forma mas detallada. No es que sean metricas distintas sino que se separaron para poder analizar mejor el cambio entre el texto original y el simplificado.

La idea fue que no bastaba con decir el simplificado tuvo cierto puntaje, sino tambien revisar si ese puntaje mejoro o no respecto al texto de entrada.

### BERTScore

Se metio porque da una medida de similitud semantica mas profunda que BLEU o ROUGE. Mientras BLEU y ROUGE revisan mas el traslape de palabras o secuencias, BERTScore ayuda a ver si el significado se mantiene aunque cambien las palabras. Por eso fue muy importante para este proyecto, ya que una simplificacion no debia perder la idea principal del texto. En el notebook de exploracion y en `metrics.py` aparece como `bertscore_f1`.

### SBERT

Tambien aparece en el script como `sbert_similarity`, pero quedo mas como metrica complementaria u opcional. Sirve tambien para similitud semantica, aunque no fue la que se tomo como principal para el ranking final.

## cuales fueron las metricas lead

Aunque se calcularon varias metricas, no todas se usaron como metricas principales para elegir configuraciones.

Las 2 metricas lead que se definieron fueron

- SARI
- BERTScore F1

Esto se ve directamente en el notebook `09_comparacion_hiperparametros`, donde el ranking final se hizo con una formula llamada `leader_score`

`leader_score = 0.6 * sari + 0.4 * (bertscore_f1 * 100)`

Eso quiere decir que para escoger las mejores configuraciones se le dio mas peso a SARI, pero tambien se tomo en cuenta BERTScore F1.

## por que se eligieron esas 2 como lead

Se eligieron esas dos porque entre todas las metricas eran las que mejor representaban el equilibrio que se queria en el trabajo.

### SARI como lead

Se eligio porque mide directamente operaciones de simplificacion. No solo pregunta si el texto se parece a la referencia sino si realmente esta haciendo cambios utiles de simplificacion. Como FEINA tambien lo usa como metrica importante, servia tanto para comparar con el paper como para evaluar la calidad de simplificacion.

### BERTScore F1 como lead

Se eligio porque ayudaba a vigilar que no se perdiera el significado original. Una salida podia tener buen SARI pero deformar la idea del texto. BERTScore ayudo a controlar eso. Por eso se decidio combinar una metrica mas enfocada a simplificacion con una metrica mas enfocada a preservacion semantica.

## para que quedaron las demas metricas

Las demas metricas no se ignoraron. Quedaron como metricas de apoyo para interpretar mejor los resultados.

Por ejemplo

- BLEU y ROUGE ayudaban a ver parecido con la referencia
- Fernandez Huerta e INFLESZ ayudaban a revisar legibilidad
- compression ratio, sentence splits, Levenshtein, exact copy, additions y deletions ayudaban a entender como cambiaba el texto
- SBERT quedaba como otra vista de similitud semantica

Entonces las metricas lead no sustituyen a las demas. Solo fueron las principales para el ranking y la toma de decisiones. Las otras siguieron sirviendo para analizar los resultados con mas detalle.
