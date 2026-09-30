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

3. **Arquitectura de Fusión Tardía (*Late Fusion*):**
   * **Extractores:** Instanciar dos ramas DenseNet-121. Cargar `densenet_frontal_best.pth` en la Rama 1 y `densenet_lateral_best.pth` en la Rama 2.
   * **Congelación Inicial:** Durante las primeras épocas, congelar las capas convolucionales (fijar `requires_grad = False`) para entrenar solo la cabeza de fusión y evitar la destrucción catastrófica de los pesos.
   * **Fusión de Vectores:** Extraer los vectores latentes finales de 1024 dimensiones de cada rama y concatenarlos:
     $$\mathbf{z}_{\text{fusion}} = [\mathbf{z}_{\text{frontal}} \,\Vert{}\, \mathbf{z}_{\text{lateral}}] \quad \rightarrow \quad \text{Vector de 2048 dimensiones}$$
   * **Clasificador Final:** Implementar un Perceptrón Multicapa (MLP) o un mecanismo de Atención Cruzada (*Cross-Attention*) que reciba el vector de 2048 dimensiones y devuelva los 14 *logits* finales.

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
   * El código debe mantener una política parametrizada y editable (ej. `uncertainty_policy='U-Ones'`) para gestionar dinámicamente si los `-1.0` se tratan como 1, 0, o se ignoran en el cálculo de la pérdida[cite: 2].

---

## 5. Estándares de Ingeniería de Software Exigidos al Agente

El agente actuará como un Ingeniero de Machine Learning Senior. Debe adherirse a las siguientes directrices operativas sin excepción:

* **Arquitectura de Software Modular (Archivos a generar/modificar):**
  * `dataset_paired.py`: Debe contener la clase `PairedCheXpertDataset` optimizada para buscar los pares Frontal/Lateral en el CSV y extraerlos del ZIP en tiempo de ejecución.
  * `models_fusion.py`: Debe contener la clase `DualStreamDenseNet` (incluyendo la lógica de carga de pesos preentrenados y la cabeza de fusión).
  * `transforms_dual.py`: Pipelines separados de *Data Augmentation* `transform_frontal` y `transform_lateral` para garantizar asincronía espacial[cite: 2].
  * `train_dual.py`: Script principal de orquestación, entrenamiento multimodal y guardado de métricas.
* **Gestión de Memoria y GPU (Crítico para Kaggle):**
  * El entrenamiento bimodal duplica el consumo de VRAM. El agente **debe** configurar hiperparámetros defensivos: reducir el `batch_size` a 16 o usar **Acumulación de Gradientes** (*Gradient Accumulation*) para simular lotes de 32 sin provocar errores *Out Of Memory* (OOM).
  * Uso estricto de `with torch.no_grad():` en todos los bucles de validación.
  * Liberación manual de memoria al final de cada época (`torch.cuda.empty_cache()`).
* **Protocolo de Interacción:**
  * El agente **NO DEBE** sobrescribir archivos complejos en un solo bloque gigante sin consultar. 
  * Debe explicar brevemente la lógica matemática o arquitectónica antes de proporcionar los bloques de código Python.
  * Todo el código debe incluir *Type Hints* de Python (`typing`) y docstrings explicativos.