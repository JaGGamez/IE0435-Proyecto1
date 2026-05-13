# Dataset — Descripción y limitaciones

## Recolección

Las imágenes fueron capturadas por los estudiantes del curso IE0435 (I Semestre 2026)
simulando una línea de producción. Se utilizó una hoja blanca como superficie base.
Cada estudiante aportó 30 fotografías: 15 con granos de arroz presentes (positivo)
y 15 sin arroz o con otros objetos como aros o clips (negativo).

- **Total de imágenes:** ~30 por estudiante, ~360 en el dataset consolidado del grupo
- **Dispositivo:** cámaras de smartphones personales
- **Entorno:** variado (distintas habitaciones, iluminación natural y artificial)
- **Formato original:** JPG

## Etiquetado

| Clase | Etiqueta | Descripción |
|---|---|---|
| Positivo | `1` | Imagen con uno o más granos de arroz sobre la superficie |
| Negativo | `0` | Imagen sin arroz (puede contener otros objetos o estar vacía) |

El etiquetado fue manual por convención de carpeta: imágenes en `Positivo/` → `1`,
imágenes en `Negativo/` → `0`. No se usó ninguna herramienta de anotación externa.

## Preprocesamiento aplicado

1. Conversión a escala de grises
2. Redimensionado a 128×128 píxeles (Lanczos)
3. Binarización con umbral 156: píxel > 156 → 1 (fondo), píxel ≤ 156 → 0 (objeto)
4. Aplanado a vector de 16 384 valores + etiqueta

## Limitaciones

- **Variabilidad de iluminación:** cada estudiante capturó con condiciones de luz
  distintas, lo que afecta directamente la binarización.
- **Variabilidad de cámara:** distintos smartphones con diferentes resoluciones,
  balances de blancos y ángulos de captura.
- **Umbral único:** se usó un umbral de binarización fijo (156) para todo el dataset,
  lo cual puede no ser óptimo para imágenes tomadas en condiciones muy diferentes.
- **Tamaño limitado:** ~357 muestras es un dataset pequeño para aprendizaje automático,
  lo que puede afectar la capacidad de generalización.
- **Etiquetado sin verificación cruzada:** cada estudiante etiquetó sus propias imágenes
  sin revisión externa, lo que puede introducir inconsistencias.
- **Objetos no contemplados:** el modelo no distingue tipos de objetos negativos
  (aros, clips, superficie vacía); todos son clase 0.
