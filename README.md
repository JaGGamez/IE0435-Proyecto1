# IE0435 – Proyecto 1: Detección de contaminaciones en línea de producción

Clasificador de imágenes para detectar granos de arroz (contaminación positiva) en una
superficie blanca, usando aprendizaje automático clásico.

**Autor:** Jacob González Gonga — B83417  
**Curso:** IE0435 Inteligencia Artificial — I Semestre 2026  
**Universidad de Costa Rica**

---

## Estructura del repositorio

```
├── Images/                  # Imágenes crudas (Positivo / Negativo)
├── All_csv/                 # Dataset consolidado del grupo
├── reports/                 # Informe final en PDF
├── image_processor.py       # Preprocesamiento: escala de grises, resize, binarización
├── threshold_explorer.py    # Herramienta visual para calibrar el umbral
├── export_csv.py            # Convierte matrices binarias a CSV etiquetado
├── classification.py        # Entrenamiento y comparación de modelos
├── export_model.py          # Exporta el mejor modelo en formato .joblib
├── visualize_images.py      # Visualización de matrices binarias
├── rename_positivo.py       # Renombra imágenes de la carpeta Positivo
├── B83417_Jacob_Gonzalez.joblib  # Modelo entrenado (entregable)
├── requirements.txt
├── LICENSE
├── README.md
├── DATASET.md
└── MODEL_CARD.md
```

---

## Instalación

```bash
pip install -r requirements.txt
```

Requiere Python 3.12 o superior.

---

## Cómo reproducir el entrenamiento

### 1. Preprocesar imágenes

```bash
python image_processor.py
```

Genera `binary_images.npz` con todas las matrices 128×128 binarizadas.

### 2. (Opcional) Calibrar el umbral de binarización

```bash
python threshold_explorer.py
```

Abre una ventana interactiva. Ajustar el valor de `THRESHOLD` en `image_processor.py`
según el resultado visual.

### 3. Exportar CSV etiquetado

```bash
python export_csv.py
```

Genera `dataset.csv` con 16 384 columnas de píxeles y una columna `label`.

### 4. Entrenar y comparar modelos

```bash
python classification.py
```

Entrena Decision Tree, Random Forest, Naive Bayes, KNN y SVM con GridSearchCV.
Guarda resultados en `classification_results.csv`.

### 5. Exportar el mejor modelo

```bash
python export_model.py
```

Genera `B83417_Jacob_Gonzalez.joblib` entrenado sobre el dataset completo.

---

## Inferencia con el modelo exportado

```python
import joblib
import numpy as np

model = joblib.load("B83417_Jacob_Gonzalez.joblib")

# imagen_vector: array de forma (1, 16384) con valores 0 o 1
prediccion = model.predict(imagen_vector)
# 1 = contaminación por arroz detectada
# 0 = sin contaminación por arroz
```

---

## Hardware usado

- CPU: AMD/Intel (sin GPU)
- RAM: 16 GB
- OS: Windows 11
- Tiempo de entrenamiento total (5 modelos + GridSearchCV): ~15 minutos
