# Machine Learning

Dos proyectos de Machine Learning de extremo a extremo que cubren las dos grandes
familias del aprendizaje automático: **supervisado** (clasificación) y **no
supervisado** (clustering). Cada proyecto recorre el ciclo completo de un problema
real: análisis exploratorio, preprocesado, modelado, evaluación y —en el primer
caso— despliegue de la solución en una aplicación web.

**Stack:** Python · scikit-learn · pandas · NumPy · SciPy · Optuna · statsmodels ·
Matplotlib · Seaborn · Streamlit · Jupyter

---

## 1. Clasificación supervisada — Predicción de suscripción a depósito bancario
[`ML_P1`](./ML_P1)

Modelo que predice si un cliente contratará un depósito a plazo a partir de datos de
una campaña de marketing bancario. Proyecto completo desde el dato crudo hasta una
aplicación web de inferencia.

**Qué incluye**
- **EDA** y limpieza de datos, con tratamiento de variables categóricas y desbalanceo.
- **Ingeniería de características** (p. ej. transformación de `pdays` en la variable
  derivada `contacted_before`), replicada de forma idéntica en entrenamiento e inferencia.
- **Comparación de modelos**: K-Nearest Neighbors y árboles de decisión.
- **Optimización de hiperparámetros (HPO)** con **Optuna**.
- **Selección del modelo final** y serialización con `joblib`.
- **Despliegue en Streamlit** ([`mystreamlit.py`](./ML_P1/mystreamlit.py)): formulario
  interactivo que carga el modelo entrenado y devuelve la predicción en tiempo real.

**Archivos clave**
| Archivo | Contenido |
|---|---|
| [`P1_ML_G84.ipynb`](./ML_P1/P1_ML_G84.ipynb) | Cuaderno principal: EDA, preprocesado, modelado, HPO y evaluación |
| [`P1_predicciones.ipynb`](./ML_P1/P1_predicciones.ipynb) | Generación de predicciones sobre datos nuevos |
| [`mystreamlit.py`](./ML_P1/mystreamlit.py) | Aplicación web de inferencia |

---

## 2. Aprendizaje no supervisado — Clustering de estrellas
[`ML_P2`](./ML_P2)

Análisis de agrupamiento sobre un conjunto de datos astronómico
([`stars_data.csv`](./ML_P2/stars_data.csv)), comparando tres enfoques de clustering y
evaluando la coherencia de los grupos obtenidos.

**Qué incluye**
- **EDA** y escalado de variables.
- **K-Medias**, **clustering jerárquico** y **DBSCAN**.
- **Comparación de algoritmos** y elección del número de clústeres.
- **Análisis e interpretación** de los grupos resultantes y conclusiones.

**Archivos clave**
| Archivo | Contenido |
|---|---|
| [`P2_ML_G84.ipynb`](./ML_P2/P2_ML_G84.ipynb) | Cuaderno principal: EDA, clustering y análisis de resultados |
| [`stars_data.csv`](./ML_P2/stars_data.csv) | Conjunto de datos |

---

## Competencias demostradas
- Ciclo completo de un proyecto de ML: del dato crudo al modelo en producción.
- Aprendizaje **supervisado** (clasificación) y **no supervisado** (clustering).
- Preprocesado e **ingeniería de características** consistente entre entrenamiento e inferencia.
- **Optimización de hiperparámetros** con Optuna.
- Evaluación y comparación rigurosa de modelos.
- **Despliegue** de modelos en una aplicación web con Streamlit.

## Cómo ejecutarlo
```bash
# Instalación del entorno (común a ambos proyectos)
pip install -r requirements.txt

# Proyecto 1
cd ML_P1
jupyter notebook P1_ML_G84.ipynb     # exploración y entrenamiento
streamlit run mystreamlit.py         # aplicación de predicción

# Proyecto 2
cd ../ML_P2
jupyter notebook P2_ML_G84.ipynb
```
