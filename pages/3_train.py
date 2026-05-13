import warnings

import pandas as pd
import streamlit as st

from ml_pipeline.artifacts import create_run_dir, save_dataframe, save_json
from ml_pipeline.automl_runner import FULL_MLJAR_ALGORITHMS, run_target_automl
from ml_pipeline.comparison import build_final_matrix, build_target_summary
from ml_pipeline.quality import analyze_data_quality
from ml_pipeline.tasks import infer_target_task, task_label

warnings.filterwarnings("ignore")

st.markdown("# 🏋️ Entrenar Modelos")

if st.session_state.df is None or st.session_state.target_cols is None:
    st.warning("⚠️ Carga el dataset y configura los targets primero.")
    st.stop()

df = st.session_state.df.copy()
targets = st.session_state.target_cols
target_configs = st.session_state.target_configs or {
    target: infer_target_task(df[target]) for target in targets
}

badge = '<span class="tag teal">MULTI-TARGET</span>' if len(targets) > 1 else '<span class="tag">SINGLE TARGET</span>'
st.markdown(
    f"**Backend:** `mljar-supervised` &nbsp; {badge} &nbsp; "
    f"**Targets:** `{'`, `'.join(targets)}`",
    unsafe_allow_html=True,
)
st.caption("Cada target se entrena con un AutoML independiente. Las columnas target se excluyen de las features.")

st.divider()

st.markdown("### 🎯 Configuración por target")
config_rows = []
invalid_targets = []
for target in targets:
    config = target_configs[target]
    if config["ml_task"] == "invalid":
        invalid_targets.append(target)
    config_rows.append(
        {
            "Target": target,
            "Tarea": task_label(config["ml_task"]),
            "Métrica primaria": config["primary_metric"],
            "Dirección": "↑ maximizar" if config["direction"] == "max" else "↓ minimizar",
            "Motivo": config["reason"],
        }
    )
st.dataframe(pd.DataFrame(config_rows), use_container_width=True, hide_index=True)

quality = analyze_data_quality(df, targets)
with st.expander("Calidad de datos antes de entrenar", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{quality['shape']['rows']:,}")
    c2.metric("Columnas", quality["shape"]["columns"])
    c3.metric("Nulos", quality["total_nulls"])
    c4.metric("Duplicados", quality["duplicate_rows"])
    if quality["target_issues"]:
        st.error("Hay targets que no se pueden entrenar.")
        st.dataframe(pd.DataFrame(quality["target_issues"]), use_container_width=True, hide_index=True)
    st.dataframe(pd.DataFrame(quality["columns"]), use_container_width=True, hide_index=True)

if invalid_targets or quality["target_issues"]:
    st.error("Corrige los targets inválidos antes de entrenar.")
    st.stop()

st.divider()
st.markdown("### ⚙️ Opciones de entrenamiento")

col1, col2, col3, col4 = st.columns(4)
with col1:
    total_time_limit = st.slider("Tiempo por target", 30, 1800, 180, 30, format="%d s")
with col2:
    test_size = st.slider("Holdout test size", 0.1, 0.4, 0.2, 0.05)
with col3:
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
with col4:
    mode = st.selectbox("Modo mljar", ["Perform", "Explain", "Compete"], index=0)

with st.expander("Catálogo mljar", expanded=False):
    st.write(", ".join(FULL_MLJAR_ALGORITHMS))
    st.caption("Se ejecuta el catálogo completo para cada target y se conserva el reporte de mljar.")

st.divider()


def _target_manifest(result: dict) -> dict:
    keys = [
        "target",
        "config",
        "feature_cols",
        "train_rows",
        "test_rows",
        "results_path",
        "leaderboard_path",
        "predictions_path",
        "metrics_path",
        "plot_paths",
        "best_model_name",
        "best_model_type",
        "best_metric_value",
    ]
    return {key: result.get(key) for key in keys}


if st.button("🚀 Entrenar AutoML por target", use_container_width=True):
    run_id, run_path = create_run_dir()
    save_json(run_path / "quality_report.json", quality)

    progress = st.progress(0, text="Iniciando corrida AutoML...")
    target_results = {}
    errors = {}

    for idx, target in enumerate(targets, start=1):
        progress.progress((idx - 1) / len(targets), text=f"Entrenando target `{target}`...")
        try:
            target_results[target] = run_target_automl(
                df,
                target,
                target_configs[target],
                all_targets=targets,
                run_path=run_path,
                test_size=float(test_size),
                total_time_limit=int(total_time_limit),
                mode=mode,
                random_state=int(random_state),
            )
        except Exception as exc:
            errors[target] = str(exc)

    progress.progress(1.0, text="Finalizando artefactos...")

    if not target_results:
        save_json(run_path / "run_manifest.json", {"run_id": run_id, "errors": errors})
        st.error("No se pudo entrenar ningún target.")
        st.json(errors)
        st.stop()

    compare_df = build_final_matrix(target_results)
    summary_df = build_target_summary(target_results)
    save_dataframe(run_path / "final_matrix.csv", compare_df.reset_index())
    save_dataframe(run_path / "target_summary.csv", summary_df)

    run_manifest = {
        "run_id": run_id,
        "base_path": str(run_path),
        "targets": {target: _target_manifest(result) for target, result in target_results.items()},
        "errors": errors,
        "settings": {
            "total_time_limit": int(total_time_limit),
            "test_size": float(test_size),
            "random_state": int(random_state),
            "mode": mode,
            "algorithms": FULL_MLJAR_ALGORITHMS,
        },
        "quality_report_path": str(run_path / "quality_report.json"),
        "final_matrix_path": str(run_path / "final_matrix.csv"),
        "target_summary_path": str(run_path / "target_summary.csv"),
    }
    save_json(run_path / "run_manifest.json", run_manifest)

    st.session_state.automl_run = {
        **run_manifest,
        "target_results": target_results,
        "compare_df": compare_df,
        "summary_df": summary_df,
    }
    st.session_state.trained_models = target_results
    st.session_state.best_model = {
        target: result.get("best_model_name") for target, result in target_results.items()
    }
    st.session_state.compare_df = compare_df

    if errors:
        st.warning("Algunos targets fallaron; los resultados válidos se conservaron.")
        st.json(errors)
    st.success(f"✅ Corrida `{run_id}` completada. Artefactos: `{run_path}`")
    st.markdown("### Tabla final target × tipo de modelo")
    st.dataframe(compare_df.round(4), use_container_width=True)
    st.markdown("### Mejor modelo por target según holdout real")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

elif st.session_state.automl_run:
    run = st.session_state.automl_run
    st.success(f"✅ Corrida AutoML disponible: `{run['run_id']}`")
    st.caption(f"Artefactos: `{run['base_path']}`")
    st.markdown("### Tabla final target × tipo de modelo")
    st.dataframe(run["compare_df"].round(4), use_container_width=True)
    st.markdown("### Mejor modelo por target según holdout real")
    st.dataframe(run["summary_df"], use_container_width=True, hide_index=True)
    st.info("Ve a Compare / Evaluate / Explainability para más análisis.")
