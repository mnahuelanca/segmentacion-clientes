# Segmentación de Clientes: Análisis y Modelado

Este proyecto tiene como objetivo analizar el perfil de una base de clientes y desarrollar modelos de Machine Learning para predecir su segmentación (A, B, C o D). El enfoque principal es entender qué características (edad, profesión, nivel de gasto, etc.) definen a cada grupo.

## Estructura del Repositorio

* `index.ipynb`: Cuaderno principal con el Análisis Exploratorio de Datos (EDA) y la limpieza inicial.
* `dfSegmentado.csv`: El conjunto de datos procesado utilizado para el entrenamiento.
* `modelosGEM.ipynb`: Implementación de modelos y optimización asistida por **Google Gemini**.
* `modelosCGPT.ipynb`: Implementación de modelos y análisis asistido por **ChatGPT**.

## El Dataset
Los datos contienen información demográfica y de comportamiento de consumo, incluyendo:
* **Demografía**: Género, Edad, Estado Civil, Graduado.
* **Socioeconómico**: Profesión, Experiencia Laboral, Puntuación de Gasto.
* **Familiar**: Tamaño del núcleo familiar.
* **Variable Objetivo**: Segmento (A, B, C, D).

## Comparativa de Modelos (IA)
Una particularidad de este proyecto es la experimentación con dos asistentes de IA distintos para la construcción de los modelos:

1.  **Versión Gemini (`modelosGEM.ipynb`)**: Se enfocó en una estructura de Pipeline, optimización con `GridSearchCV` y análisis de métricas como la matriz de confusión y curvas ROC. Logró una exactitud del **51%** con un Random Forest Optimizado.
2.  **Versión ChatGPT (`modelosCGPT.ipynb`)**: Se enfocó en una implementación rápida de árboles de decisión y bosques aleatorios, evaluando el impacto de la reducción de variables en el rendimiento del modelo.

## Tecnologías
* **Python 3.x**
* **Pandas / Numpy**: Manipulación de datos.
* **Scikit-Learn**: Machine Learning y Preprocesamiento.
* **Matplotlib / Seaborn**: Visualización de datos.

## Conclusiones
El análisis permitió identificar que variables como la **Edad** y el **Nivel de Gasto** son los predictores más fuertes para determinar el segmento de un cliente. La comparación entre los modelos realizados con distintos asistentes de IA permitió evaluar diferentes enfoques de limpieza y ajuste de hiperparámetros.
