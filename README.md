# ⚗️ ML Studio — AutoML Tabular con MLJAR

Interfaz visual para investigar, entrenar, comparar, evaluar y explicar modelos de machine learning tabular con `mljar-supervised`.

## Stack

| Módulo | Librería |
|---|---|
| UI | Streamlit |
| AutoML | `mljar-supervised` |
| Modelos base | Linear, Random Forest, Extra Trees, LightGBM, Xgboost, CatBoost, Neural Network, Nearest Neighbors |
| Evaluación | scikit-learn, Plotly |
| Explainability | Reportes MLJAR, permutation importance |

## Instalación

```bash
cd ml_studio
pip install -r requirements.txt
streamlit run app.py
```

## Flujo del pipeline

```text
📂 Dataset  →  🔍 EDA  →  🏋️ Train  →  📊 Compare  →  🔬 Evaluate  →  🚨 Early Warning  →  🧠 Explain
```

### 1. 📂 Dataset
- Sube un CSV/Excel o usa un dataset de ejemplo.
- Selecciona una o varias variables objetivo.
- La app infiere clasificación binaria, clasificación multiclase o regresión por target.
- Puedes corregir manualmente la tarea inferida antes de entrenar.

### 2. 🔍 EDA & Calidad
- Resumen de tipos, nulos, duplicados, distribuciones, correlaciones y outliers.
- Pestaña **📐 Estadísticas** con análisis detallado por variable:
  - **Mínimo, máximo, media, mediana, varianza** y **test de Shapiro-Wilk** (α = 0.05).
  - Sub-muestreo reproducible a 5000 observaciones si el dataset supera el límite de SciPy.
  - Selector de α y filtro *solo numéricas*.
  - Descarga del resumen en **CSV** (formato ancho) y **JSON** (completo) desde la propia pestaña.
- El entrenamiento guarda un `quality_report.json` por corrida.

### 3. 🏋️ Entrenar
- Ejecuta un `AutoML` independiente por cada target.
- Excluye todos los targets de las features para evitar leakage.
- Usa holdout externo reproducible para evaluar el mejor modelo por target.
- Conserva el reporte completo de MLJAR en `artifacts/automl_runs/{run_id}/{target}/mljar/`.

### 📦 Cargar corridas guardadas
- En la sección **Train Models** puedes seleccionar una corrida ya entrenada en `artifacts/automl_runs/`.
- Compare / Evaluate / Explainability se alimentan de los artefactos guardados sin re-entrenar.
- Nota: las secciones que requieren el modelo en memoria (permutation importance y predicción manual)
  quedan deshabilitadas cuando la corrida se carga desde disco.

### 4. 📊 Comparar
- Muestra la tabla final target × tipo de modelo.
- Cada celda contiene el mejor valor de la métrica primaria para ese target y tipo de modelo.
- Clasificación usa `f1` y se maximiza; regresión usa `rmse` y se minimiza.

### 5. 🔬 Evaluar
- Clasificación: accuracy, F1, precision, recall, matriz de confusión y curvas binarias cuando hay probabilidades.
- Regresión: score global compuesto, R², R² ajustado, MAE, RMSE, MAPE, SMAPE, real vs predicho, residuos y distribución de error.

### 6. 🚨 Early Warning
- Disponible para targets de regresión con especificaciones en `config/quality_specs.json`.
- Usa residuos de calibración para calcular P(DZ) y P(OOS) por batch.
- Muestra alertas, métricas de clasificación, análisis de umbrales y distribución de riesgo.

### 7. 🧠 Explainability
- Abre artefactos explicativos del reporte MLJAR.
- Calcula permutation importance sobre el mejor AutoML por target.
- Permite predicción manual para el target seleccionado.

## Artefactos

Cada corrida crea:

```text
artifacts/automl_runs/{run_id}/
  run_manifest.json
  quality_report.json
  final_matrix.csv
  target_summary.csv
  {target}/
    target_manifest.json
    leaderboard.csv
    holdout_metrics.json
    predictions.csv
    calibration_residuals.csv
    early_warning_predictions.csv
    early_warning_metrics.json
    plots/*.html
    mljar/
```

## Datasets de ejemplo incluidos

| Dataset | Targets sugeridos |
|---|---|
| 🏠 Housing | `price_eur`, `rent_eur`, `quality_score` |
| 🌸 Iris | `target` |
| 💳 Credit Risk | `default`, `fraud_flag` |
| 🌡️ Energy | `heating_load`, `cooling_load` |
| 🍷 Wine Quality | `target` |
