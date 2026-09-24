# AGENTS.md — Contexto Operativo y Directrices del Proyecto

## 1. Identificación y Misión del Proyecto
* **Referencia:** SV-26-GIJON-1-14[cite: 1]
* **Investigador Responsable:** Ángel Francisco del Río Álvarez[cite: 1]
* **Título Oficial:** Detección y generalización multi-proyección de nódulos pulmonares en radiografías de tórax mediante aprendizaje profundo y segmentación anatómica (DL-LUNGSCREENING)[cite: 1].
* **Objetivo General:** Desarrollar un sistema de Deep Learning capaz de detectar nódulos pulmonares combinando radiografías de tórax frontales (PA/AP) y laterales (LAT) mediante una arquitectura de fusión tardía (*Late Fusion*)[cite: 1]. La vista lateral es crítica para resolver diagnósticos en regiones anatómicas con superposición ósea o cardíaca (zonas retrocardíacas, retroesternales y ángulos costofrénicos posteriores).

---

## 2. Marco Metodológico y Justificación de Datos

El objetivo final es la **detección de nódulos**, no predecir directamente "cáncer" (lo cual requiere biopsia o TC de alta resolución)[cite: 1]. Una radiografía solo revela la opacidad/nódulo. Para alcanzar este fin, el proyecto sigue un flujo en tres fases:

### Fase 1: Pre-entrenamiento en Dominio Médico con CheXpert (FASE ACTUAL)
* **Propósito:** Las redes preentrenadas en ImageNet no comprenden la radiología torácica. Entrenar sobre CheXpert enseña al *backbone* la anatomía del tórax y el diagnóstico diferencial de densidades pulmonares.
* **14 Clases Completas:** No se usa el subconjunto de 5 clases porque omite el nódulo. Se entrena explícitamente con las 14 observaciones de CheXpert para capturar **`Lung Lesion`** (nódulos/masas) y **`Lung Opacity`** (manifestación visual del nódulo), junto al resto de patologías:
  `['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']`.
* **Modelos Univariantes (Baselines Independientes):**
  * `Baseline Frontal`: Entrenado y evaluado exclusivamente con proyecciones frontales (`Frontal/Lateral == 'Frontal'`).
  * `Baseline Lateral`: Entrenado y evaluado exclusivamente con proyecciones laterales (`Frontal/Lateral == 'Lateral'`).
  * **Regla estricta:** No mezclar vistas en los *baselines*. El filtrado se realiza por software (`train.csv` / `valid.csv`), sin duplicar ni mover archivos en disco.

### Fase 2: Fusión Multimodal (Late Fusion)
* Se toman los dos *encoders* DenseNet-121 preentrenados en la Fase 1.
* Se extraen los vectores latentes (1024 características cada uno) y se concatenan:
  $$\mathbf{z}_{\text{fusion}} = [\mathbf{z}_{\text{frontal}} \,\Vert{}\, \mathbf{z}_{\text{lateral}}] \quad (2048 \text{ dimensiones})$$
* Un clasificador conjunto (*MLP*) aprende a correlacionar ambas proyecciones para generar la predicción multietiqueta final.
* Se realiza *fine-tuning* progresivo de las capas superiores con bajo *learning rate*.

### Fase 3: Especialización en Nódulos (FASE FUTURA / BLOQUEADA)
* Enriquecimiento y evaluación final orientada a la detección/segmentación fina de nódulos y generalización anatómica con el dataset sintético[cite: 1].

---

## 3. Protocolos de Datos y Reglas de Bloqueo

> ⛔ **BLOQUEO ABSOLUTO — ARCHIVO ZENODO:**
> Existe en el entorno un archivo comprimido denominado `Geometric_lung_rx...` (datos sintéticos de Zenodo).
> **QUEDA ESTRICTAMENTE PROHIBIDO ABRIR, DESCOMPRIMIR, INSPECCIONAR O CARGAR ESTE ARCHIVO.**
> Su uso está reservado en exclusiva para la Fase 3 del proyecto. Cualquier interacción con él en esta etapa invalidará la metodología.

* **Dataset Operativo:** `archive.zip` (~11 GB) junto a sus metadatos `train.csv` y `valid.csv`.
* **Carga de Imágenes:** Lectura bajo demanda (*lazy loading*) desde el `.zip` directamente a través de `torch.utils.data.Dataset`.
* **Manejo de Rutas:** Filtrar siempre archivos residuales generados por sistemas tipo UNIX/macOS (prefijo `._`).

---

## 4. Decisiones Técnicas y Matemáticas de Consenso

1. **Tratamiento de Incertidumbre (`-1.0`):**
   * Política adoptada: **Pendiente de decisión**. Aún no se ha determinado si se usará `U-Ones`, `U-Zeros` o `U-Ignore` para los valores `-1.0` y ausentes (`NaN`).
   * El código del dataset (`dataset.py`) debe mantener esta política parametrizada y completamente editable (ej. `uncertainty_policy='U-Ones'`) hasta que se tome la decisión final.
2. **Función de Pérdida (*Loss Function*):**
   * **`BCEWithLogitsLoss` con balanceo dinámico mediante `pos_weight`**.
   * Para cada una de las 14 clases $c$, el peso se calcula dinámicamente sobre el subconjunto de entrenamiento según:
     $$\text{pos\_weight}_c = \frac{N_{\text{negativos}, c}}{N_{\text{positivos}, c}}$$
   * No se usará *Focal Loss* en la etapa inicial para evitar inestabilidad por hiperparámetros.
3. **Métrica de Evaluación:**
   * **AUROC (Área bajo la curva ROC)** calculada con `scikit-learn`.
   * Se reportará el AUROC individual de cada una de las 14 clases y el promedio macro.
4. **Arquitectura y Adaptación:**
   * **DenseNet-121** preentrenada en ImageNet (`DenseNet121_Weights.DEFAULT`).
   * Adaptación de la primera convolución (`conv0`) de 3 canales a **1 canal (escala de grises médica)** mediante el promedio de los pesos de los canales RGB originales.
   * Capa clasificadora final reemplazada por una capa lineal de **14 salidas**.

---

## 5. Estándares de Ingeniería de Software para el Asistente

El agente debe trabajar bajo estándares de desarrollo sénior en Python y PyTorch:
* **Estructura Modular:** El código debe dividirse en módulos especializados:
  * `dataset.py`: Definición de `CheXpertDataset` con lectura en memoria volátil desde el ZIP.
  * `models.py`: Arquitectura DenseNet-121 monocanal y cabezas de fusión.
  * `transforms.py`: Pipelines de normalización y data augmentation.
  * `utils.py`: Cálculo de pesos de pérdida y métricas (AUROC).
  * `train.py`: Bucles de entrenamiento, validación y gestión de checkpoints.
* **Consulta Previa Obligatoria:** Consultar cualquier modificación estructural, cambio de hiperparámetros de optimización o inclusión de dependencias adicionales antes de su ejecución.
* **Tipado y Documentación:** Uso estricto de *type hints* (`typing`) y docstrings descriptivos en cada función y clase.
* **Eficiencia:** Control exhaustivo de VRAM en GPU, uso de tensores sin acumulación de grafos en validación (`torch.no_grad()`) y optimización de workers en `DataLoader`.