# AGENTS.md — Contexto Operativo y Directrices Estrictas del Proyecto

## 1. Identificación y Misión del Proyecto
* **Referencia:** SV-26-GIJON-1-14[cite: 2]
* **Investigador Responsable:** Ángel Francisco del Río Álvarez[cite: 2]
* **Título Oficial:** Detección y generalización multi-proyección de nódulos pulmonares en radiografías de tórax mediante aprendizaje profundo y segmentación anatómica (DL-LUNGSCREENING)[cite: 2].
* **Objetivo General Clínico:** Desarrollar un sistema de Deep Learning capaz de detectar nódulos pulmonares y otras anomalías combinando radiografías de tórax frontales (PA/AP) y laterales (LAT)[cite: 2]. 
* **Problema Fundamental a Resolver (Asincronía Clínica):** Las tomas radiográficas no son simultáneas. Entre la toma frontal y lateral, el paciente cambia de postura, respira de forma distinta y se reposiciona. El sistema **debe** aprender a correlacionar hallazgos anatómicos en 3D tolerando estas desalineaciones e inconsistencias espaciales.

---

## 2. Marco Metodológico y Estado Actual del Proyecto

El objetivo es la **detección de nódulos / opacidades** evaluando simultáneamente las 14 clases de CheXpert (foco crítico en `Lung Lesion` y `Lung Opacity`)[cite: 2]. El proyecto consta de tres fases. Las instrucciones para el agente varían según la fase.

### Fase 1: Pre-entrenamiento Univariante de Especialistas (Fase COMPLETADA - CONGELADA)
Se han entrenado dos *backbones* DenseNet-121 independientes. **Regla estricta:** El agente NO debe sugerir reentrenar estos modelos desde cero.
* **Especialista Frontal:** Entrenado con ~190k imágenes. Mejor modelo guardado como `densenet_frontal_best.pth` (Macro AUROC: 0.8088).
* **Especialista Lateral:** Entrenado con ~30k imágenes. Mejor modelo guardado como `densenet_lateral_best.pth` (Macro AUROC: 0.7627).
* **Uso en siguientes fases:** Estos archivos `.pth` se utilizarán exclusivamente como inicializadores de pesos (extractores de características) para la Fase 2.

### Fase 2: Arquitectura Bimodal / Fusión Multimodal (Fase ACTUAL - EN DESARROLLO)
El agente debe guiar el código hacia la construcción de un modelo "Dual-Stream" (dos flujos) que procese ambas vistas de forma simultánea. Se deben implementar los siguientes pilares sin desviaciones:

1. **Filtrado Estricto de Datos (Estudios Pareados):**
   * El dataset debe agruparse por `patient_id` y `study_id`.
   * **Regla de Exclusión:** Cualquier estudio que no contenga al menos UNA vista Frontal y UNA vista Lateral debe ser descartado del entrenamiento bimodal.
   * La salida del `DataLoader` debe ser siempre una tupla: `(tensor_frontal, tensor_lateral, etiquetas_del_estudio)`.

2. **Simulación de Asincronía mediante Data Augmentation Independiente:**
   * Para enseñar a la red que el paciente se mueve entre tomas, las transformaciones geométricas **NO deben aplicarse con la misma semilla** a ambas imágenes.
   * Se deben aplicar rotaciones (±5° a ±7°), traslaciones sutiles y variaciones de escala de forma completamente **independiente** al tensor frontal y al tensor lateral en cada iteración de entrenamiento.

3. **Arquitectura SOTA de Fusión Bimodal:**
   * **Extractores:** Instanciar dos ramas DenseNet-121. Cargar `densenet_frontal_best.pth` en la Rama 1 y `densenet_lateral_best.pth` en la Rama 2.
   * **Atención Cruzada (Cross-Attention):** En lugar de una simple concatenación plana y un MLP, el modelo debe utilizar un mecanismo de *Multi-Head Cross-Attention* en el "Cerebro". Esto permite que el vector Frontal consulte proactivamente características específicas del vector Lateral (y viceversa) para alinear la información anatómica desfasada de forma inteligente antes de emitir los 14 logits.
   * **Entrenamiento en 2 Fases (Descongelación Gradual):**
     * *Fase Warm-up:* Durante las primeras épocas (ej. 3-4), congelar los backbones (`requires_grad = False`) y entrenar solo el mecanismo de Atención con un *Learning Rate* normal (ej. `1e-4`) para asentar la lógica de cruce de datos.
     * *Fase Fine-Tuning End-to-End:* Descongelar los backbones al completo y seguir entrenando toda la macro-red con un *Learning Rate* muy bajo (ej. `1e-5`) para que los extractores aprendan características sinérgicas.
   * **Optimizadores (LR Schedulers):** Es obligatorio integrar atenuadores de aprendizaje como `ReduceLROnPlateau` o `CosineAnnealingLR` para exprimir las métricas AUROC en la fase de convergencia final.

### Fase 3: Especialización en Nódulos (Fase FUTURA - ESTRICTAMENTE BLOQUEADA)
* Enriquecimiento y evaluación final orientada a la detección/segmentación fina de nódulos y generalización anatómica con el dataset sintético[cite: 2].

---

## 3. Protocolos de Datos y Reglas de Bloqueo Inquebrantables

> ⛔ **BLOQUEO ABSOLUTO — ARCHIVO ZENODO (DATOS SINTÉTICOS):**
> Existe en el entorno un archivo comprimido denominado `Geometric_lung_rx...` (datos sintéticos de Zenodo).
> **QUEDA ESTRICTAMENTE PROHIBIDO ABRIR, DESCOMPRIMIR, INSPECCIONAR, CARGAR O HACER REFERENCIA EN EL CÓDIGO A ESTE ARCHIVO.**
> Su uso está reservado en exclusiva para la Fase 3 del proyecto[cite: 2]. Cualquier intento del agente de incluir este archivo en los scripts actuales será considerado un fallo crítico.

* **Entorno de Trabajo:** Plataforma Kaggle (`/kaggle/working/`).
* **Origen de Datos Obligatorios:** 
  * Imágenes (Lectura *lazy* directa desde ZIP): `/kaggle/input/datasets/ashery/chexpert`[cite: 2]
  * Código base y divisiones (CSV): `/kaggle/input/datasets/eloysuarezcostales/lungscreen`

---

## 4. Decisiones Técnicas y Matemáticas de Consenso (No Modificables)

1. **Métrica de Evaluación Principal:**
   * **Macro AUROC** (Área bajo la curva ROC promedio de las 14 clases).
   * Queda estrictamente prohibido usar *Accuracy* (precisión global) como métrica de Early Stopping o selección de modelo debido al desbalanceo de clases médicas.
2. **Función de Pérdida (*Loss Function*):**
   * Uso obligatorio de **`BCEWithLogitsLoss` con balanceo dinámico mediante `pos_weight`**[cite: 2].
   * Fórmula de balanceo implementada en código:
     $$\text{pos\_weight}_c = \frac{N_{\text{negativos}, c}}{N_{\text{positivos}, c}}$$
3. **Tratamiento de Incertidumbre (Etiquetas `-1.0`):**
   * El código debe mantener una política parametrizada y editable (ej. `uncertainty_policy='U-Ignore'`) para gestionar dinámicamente si los `-1.0` se tratan como 1, 0, o se ignoran.
   * **Implementación Matemática Estricta de `U-Ignore`**: Cuando esta política está activa, un diagnóstico incierto (`-1.0` o nulo) **nunca debe descartar el estudio completo** (ya que otras patologías en esa misma radiografía pueden ser útiles). Para ello:
     1. **Entrenamiento (Loss Masking):** Se debe aplicar una máscara dinámica (`valid_mask = (labels != -1.0).float()`) que multiplique el error de esa predicción por `0`. Esto exige instanciar la pérdida de PyTorch con `reduction='none'` para que el sistema ni castigue ni premie los aciertos/fallos sobre etiquetas inciertas.
     2. **Validación (Métricas):** El cálculo final del rendimiento debe usar un filtro previo (ej. dentro de `compute_auroc`) que extraiga y descarte los pares `(predicción, etiqueta_real)` marcados como `-1.0` antes de inyectarlos en el motor estadístico del AUROC. Esto asegura una evaluación puramente justa.

---

## 5. Estándares de Ingeniería de Software Exigidos al Agente

El agente actuará como un Ingeniero de Machine Learning Senior. Debe adherirse a las siguientes directrices operativas sin excepción:

* **Arquitectura de Software Modular (Archivos del Proyecto - Estado Actual SOTA):**
  * `dataset_paired.py`: Implementa la clase `PairedCheXpertDataset`. Se encarga de la lectura eficiente de imágenes emparejadas (Frontal/Lateral) directamente desde el archivo `.zip`. Administra el parámetro `uncertainty_policy` para lidiar estructuralmente con los valores inciertos.
  * `models_fusion.py`: Implementa la arquitectura dual `DualStreamDenseNet`.
    - Aloja las dos ramas extractoras congelables inicializadas con pesos de la Fase 1.
    - Implementa `CrossAttentionFusion`, un mecanismo basado en `nn.MultiheadAttention` donde el vector Frontal y el Lateral se consultan bidireccionalmente para suplir las desalineaciones espaciales.
    - Contiene el método `unfreeze_backbones()` encargado de abrir las compuertas para el Fine-Tuning de todo el sistema.
  * `transforms_dual.py`: Pipelines de *Data Augmentation* separados (`get_transforms_frontal` y `get_transforms_lateral`) para aplicar transformaciones con distintas magnitudes (ej. ±5° frontal vs ±7° lateral), forzando al modelo a aprender generalización 3D sin depender de un paciente perfectamente quieto.
  * `train_dual.py`: Script avanzado de orquestación del entrenamiento Bimodal. Responsabilidades clave:
    - **Enmascaramiento Matemático (U-Ignore)**: Implementa máscaras lógicas multiplicadas por 0 en la `BCEWithLogitsLoss` (`reduction='none'`) para que el modelo no sea castigado ni premiado al evaluar una enfermedad dudosa (`-1.0`), manteniendo intactas las demás enfermedades de la radiografía.
    - **Evaluación AUROC Dinámica**: En la validación usa `compute_auroc()` para purgar las etiquetas `-1.0` antes del cálculo estadístico.
    - **Descongelación Gradual (Warm-up a Fine-Tuning)**: Entrena solo el Cross-Attention durante las épocas iniciales (LR `1e-4`). Al alcanzar `warmup_epochs`, descongela toda la red automáticamente y reinicia el optimizador con un LR microscópico (LR `1e-5`) para ajustar finamente sin destruir el conocimiento previo.
    - **Learning Rate Scheduler**: Utiliza `ReduceLROnPlateau` para mitigar atascos en la convergencia reduciendo la tasa a la mitad de forma dinámica.
    - **Seguridad Memoria (OOM)**: Usa Acumulación de Gradientes (`GRADIENT_ACCUMULATION_STEPS`) y limpia explícitamente la VRAM en cada época.
* **Protocolo de Interacción:**
  * El agente **NO DEBE** sobrescribir archivos complejos en un solo bloque gigante sin consultar. 
  * Debe explicar brevemente la lógica matemática o arquitectónica antes de proporcionar los bloques de código Python.
  * Todo el código debe incluir *Type Hints* de Python (`typing`) y docstrings explicativos.