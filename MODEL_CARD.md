# Model Card — B83417_Jacob_Gonzalez

## Model name + version

**Rice Contamination Classifier v1.0**  
Archivo: `B83417_Jacob_Gonzalez.joblib`  
Autor: Jacob González Gonga — B83417  
Fecha: Mayo 2026

---

## Intended use

**Uso previsto:**  
Detectar si una imagen de una superficie de producción contiene granos de arroz
(contaminación positiva), como apoyo a sistemas de inspección visual automatizada.

**Fuera de alcance:**
- No debe usarse en entornos de producción real sin validación adicional
- No distingue tipos de contaminación (solo arroz vs. no arroz)
- No es apto para imágenes con fondos distintos a una superficie blanca
- No generaliza a objetos fuera del contexto del dataset de entrenamiento

---

## Data summary

- **Origen:** imágenes capturadas por estudiantes del curso IE0435 (UCR, I-2026)
- **Tamaño:** 357 muestras (177 negativas, 180 positivas)
- **Formato de entrada al modelo:** vector binario de 16 384 valores (imagen 128×128 binarizada)
- **Variaciones presentes:** distintas condiciones de iluminación (natural/artificial),
  diferentes smartphones y ángulos de captura
- **Partición:** 80% entrenamiento (285 muestras), 20% prueba (72 muestras), estratificada

---

## Labeling process

- **Método:** etiquetado por convención de carpeta (Positivo → 1, Negativo → 0)
- **Herramienta:** ninguna automatizada; clasificación manual por cada estudiante
- **Calidad:** no se realizó verificación cruzada entre estudiantes
- **Consistencia:** el umbral de binarización (156) fue determinado visualmente
  con la herramienta `threshold_explorer.py`, lo que introduce subjetividad

---

## Metrics

| Métrica | Valor |
|---|---|
| Accuracy | 0.736 |
| Precision | 0.667 |
| Recall | 0.944 |
| F1-score | 0.782 |
| F1 validación cruzada (5-fold) | 0.742 |

**Conjunto de prueba:** 72 muestras (36 positivas, 36 negativas)  
**Métrica de selección:** F1-score (balance entre precisión y recall)  
**Matriz de confusión:**

|  | Predicho Negativo | Predicho Positivo |
|---|---|---|
| **Real Negativo** | 19 | 17 |
| **Real Positivo** | 2 | 34 |

El modelo detecta el 94.4% de las contaminaciones reales con solo 2 falsos negativos.

---

## Ethical / safety notes

- **Sesgo por iluminación:** imágenes tomadas con poca luz tienden a binarizarse
  de forma incorrecta, generando vectores ruidosos que el modelo no puede interpretar bien.
- **Sesgo por dispositivo:** la calidad de la cámara afecta el contraste y
  la nitidez del arroz, lo que puede provocar clasificaciones erróneas.
- **Sesgo por fondo:** el modelo asume fondo blanco uniforme. Cualquier mancha,
  sombra o irregularidad en la superficie puede ser interpretada como contaminación.
- **No apto para decisiones críticas** sin supervisión humana adicional.

---

## Limitations

- El dataset es pequeño (~357 muestras), lo que limita la capacidad de generalización.
- El umbral de binarización fijo (156) no se adapta a variaciones de iluminación.
- Objetos muy pequeños o parcialmente ocultos pueden no generar suficientes píxeles
  oscuros para ser detectados.
- Imágenes con desenfoque (blur) pierden los bordes del arroz y pueden clasificarse
  incorrectamente como negativas.
- El modelo no fue probado con imágenes de cámaras industriales o condiciones
  controladas de iluminación.

---

## Reproducibility

**Instalación:**
```bash
pip install -r requirements.txt
```

**Entrenamiento completo:**
```bash
python image_processor.py
python export_csv.py
python classification.py
python export_model.py
```

**Hiperparámetros del modelo final:**
- Algoritmo: Random Forest
- `criterion`: entropy
- `max_depth`: 10
- `n_estimators`: 200
- `random_state`: 42
- Preprocesamiento: VarianceThreshold(threshold=0.0)

**Hardware usado:**
- CPU: procesador de uso general (sin GPU)
- RAM: 16 GB
- OS: Windows 11
- Python: 3.12.6
- Tiempo de entrenamiento (export_model.py): ~30 segundos
