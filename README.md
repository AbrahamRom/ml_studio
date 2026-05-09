# ⚗️ ML Studio — Pipeline Visual de Machine Learning

Interfaz visual completa para entrenar, comparar y explicar modelos de ML con soporte **multioutput**.

## Stack

| Módulo | Librería |
|---|---|
| UI | Streamlit |
| Modelos | scikit-learn, XGBoost, LightGBM |
| Multioutput | `MultiOutputClassifier` / `MultiOutputRegressor` |
| Explainabilidad | SHAP TreeExplainer, Permutation Importance, PDP |
| Visualización | Plotly |

## Instalación

```bash
# 1. Clona / copia la carpeta ml_studio
cd ml_studio

# 2. Instala dependencias
pip install -r requirements.txt

# 3. Lanza la app
streamlit run app.py
```

## Flujo del pipeline

```
📂 Dataset  →  🔍 EDA  →  🏋️ Train  →  📊 Compare  →  🔬 Evaluate  →  🧠 Explain
```

### 1. 📂 Dataset
- Sube tu propio CSV / Excel **o** usa uno de los 5 datasets de ejemplo
- Selecciona múltiples columnas objetivo → activa modo **MULTIOUTPUT**
- Elige tarea: `classification` o `regression`

### 2. 🔍 EDA & Calidad
- Resumen de tipos y estadísticas
- Histogramas, scatter plots, boxplots
- Matriz de correlación interactiva
- Detección de nulos y outliers (IQR)

### 3. 🏋️ Entrenar Modelos
- Selecciona cualquier combinación de modelos (LR, RF, XGBoost, LightGBM, SVM, KNN, etc.)
- Control de test size y CV folds
- En modo multioutput usa automáticamente `MultiOutputClassifier` / `MultiOutputRegressor`
- Métricas individuales por target + promedio global

### 4. 📊 Comparar Modelos
- Tabla de ranking con highlight del mejor modelo
- Gráficas de barras por métrica
- **Radar chart** normalizado para comparación visual holística
- Desglose por target individual (multioutput)

### 5. 🔬 Evaluar Modelos
- **Clasificación:** Matriz de confusión (con normalización), reporte completo, curvas ROC y PR
- **Regresión:** Real vs Predicho, análisis de residuos, Q-Q plot, histograma de errores
- Vista multioutput: métricas lado a lado para todos los targets

### 6. 🧠 Explainabilidad
- **Feature Importance nativa** (árboles) con importancia acumulada
- **Permutation Importance** (model-agnostic, funciona con cualquier modelo)
- **Partial Dependence Plots** con rug plot de distribución real
- **SHAP TreeExplainer:** beeswarm plot, waterfall por observación
- **Predicción manual:** ingresa valores custom y obtén predicción en tiempo real

## Modelos disponibles

### Clasificación
`lr` · `rf` · `xgboost` · `lightgbm` · `dt` · `knn` · `gbc` · `et` · `svm` · `nb`

### Regresión
`lr` · `ridge` · `lasso` · `rf` · `xgboost` · `lightgbm` · `gbr` · `et` · `dt` · `knn`

## Datasets de ejemplo incluidos

| Dataset | Tarea | Targets |
|---|---|---|
| 🏠 Housing | Regresión multioutput | `price_eur`, `rent_eur`, `quality_score` |
| 🌸 Iris | Clasificación | `target` |
| 💳 Credit Risk | Clasificación multioutput | `default`, `fraud_flag` |
| 🌡️ Energy | Regresión multioutput | `heating_load`, `cooling_load` |
| 🍷 Wine Quality | Regresión | `target` |
