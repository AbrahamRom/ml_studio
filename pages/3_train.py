import warnings

import pandas as pd
import streamlit as st

from ml_pipeline.artifacts import create_run_dir, save_dataframe, save_json
from ml_pipeline.automl_runner import FULL_MLJAR_ALGORITHMS, run_target_automl
from ml_pipeline.comparison import (
    build_best_model_metrics,
    build_final_matrix,
    build_target_summary,
)
from ml_pipeline.quality import analyze_data_quality
from ml_pipeline.tasks import infer_target_task, normalize_target_config, task_label
from utils.pagination import paginated_dataframe

warnings.filterwarnings("ignore")

st.markdown("# 🏋️ Entrenar Modelos")

if st.session_state.df is None:
    st.warning("⚠️ Carga el dataset primero.")
    st.stop()

df = st.session_state.df.copy()

tab_automl, tab_dl = st.tabs(["🤖 AutoML (mljar)", "🧠 Deep Learning"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: AutoML
# ═══════════════════════════════════════════════════════════════════════════════
with tab_automl:
    st.markdown("### 🎯 Configurar targets")
    st.caption("Cada target se entrena con un AutoML independiente. Las columnas target se excluyen de las features.")

    targets = st.multiselect(
        "Variables objetivo (target/s)",
        options=df.columns.tolist(),
        default=st.session_state.target_cols or [],
        help="Selecciona las columnas que quieres predecir.",
        key="aml_targets",
    )

    if targets:
        is_multi = len(targets) > 1
        badge = '<span class="tag teal">MULTI-TARGET</span>' if is_multi else '<span class="tag">SINGLE TARGET</span>'
        st.markdown(f"**Modo:** {badge} &nbsp; **Backend:** `mljar-supervised`", unsafe_allow_html=True)

        st.markdown("#### Tarea por target")
        task_options = ["binary_classification", "multiclass_classification", "regression"]
        target_configs = {}
        blocking_targets = []

        for target in targets:
            inferred = infer_target_task(df[target])
            if inferred["ml_task"] == "invalid":
                blocking_targets.append(target)
                st.error(f"`{target}` no es entrenable: {inferred['reason']}")
                target_configs[target] = inferred
                continue

            default_idx = task_options.index(inferred["ml_task"])
            selected_ml_task = st.selectbox(
                f"`{target}`",
                task_options,
                index=default_idx,
                format_func=task_label,
                key=f"aml_target_task_{target}",
                help=inferred["reason"],
            )

            config = normalize_target_config({**inferred, "ml_task": selected_ml_task})
            if selected_ml_task != inferred["ml_task"]:
                config["reason"] = f"Override manual. Inferencia original: {task_label(inferred['ml_task'])}."
            target_configs[target] = config

            metric_text = "maximizar" if config["direction"] == "max" else "minimizar"
            st.caption(
                f"Inferido: {task_label(inferred['ml_task'])}. "
                f"Métrica primaria: `{config['primary_metric']}` ({metric_text}). "
                f"{config['reason']}"
            )

        config_rows = []
        for target in targets:
            config = target_configs[target]
            config_rows.append({
                "Target": target,
                "Tarea": task_label(config["ml_task"]),
                "Métrica primaria": config["primary_metric"],
                "Dirección": "↑ maximizar" if config["direction"] == "max" else "↓ minimizar",
                "Motivo": config["reason"],
            })
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

        automl_invalid = bool(blocking_targets or quality["target_issues"])
        if automl_invalid:
            st.error("Corrige los targets inválidos antes de entrenar AutoML.")

        st.divider()
        st.markdown("### ⚙️ Opciones de entrenamiento")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_time_limit = st.slider("Tiempo por target", 30, 1800, 180, 30, format="%d s", key="aml_time")
        with col2:
            test_size = st.slider("Holdout test size", 0.1, 0.4, 0.2, 0.05, key="aml_test")
        with col3:
            random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1, key="aml_rs")
        with col4:
            mode = st.selectbox("Modo mljar", ["Perform", "Explain", "Compete"], index=0, key="aml_mode")

        with st.expander("Catálogo mljar", expanded=False):
            st.write(", ".join(FULL_MLJAR_ALGORITHMS))

        st.divider()

        if st.button("✅ Confirmar targets y 🚀 Entrenar AutoML", use_container_width=True, type="primary", disabled=automl_invalid):
            st.session_state.target_cols = targets
            st.session_state.target_configs = target_configs
            st.session_state.task_type = "per_target"
            st.session_state.multioutput = is_multi

            run_id, run_path = create_run_dir()
            save_json(run_path / "quality_report.json", quality)

            progress = st.progress(0, text="Iniciando corrida AutoML...")
            target_results = {}
            errors = {}

            for idx, target in enumerate(targets, start=1):
                progress.progress((idx - 1) / len(targets), text=f"Entrenando target `{target}`...")
                try:
                    target_results[target] = run_target_automl(
                        df, target, target_configs[target], all_targets=targets,
                        run_path=run_path, test_size=float(test_size),
                        total_time_limit=int(total_time_limit), mode=mode,
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
            best_metrics_df = build_best_model_metrics(target_results)
            save_dataframe(run_path / "final_matrix.csv", compare_df.reset_index())
            save_dataframe(run_path / "target_summary.csv", summary_df)
            best_metrics_path = None
            if not best_metrics_df.empty:
                best_metrics_path = save_dataframe(run_path / "best_model_metrics.csv", best_metrics_df)

            def _target_manifest(result):
                keys = ["target", "config", "feature_cols", "train_rows", "test_rows",
                        "results_path", "leaderboard_path", "predictions_path",
                        "metrics_path", "plot_paths", "best_model_name", "best_model_type",
                        "best_metric_value"]
                return {key: result.get(key) for key in keys}

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
                "best_model_metrics_path": str(best_metrics_path) if best_metrics_path else None,
            }
            save_json(run_path / "run_manifest.json", run_manifest)

            st.session_state.automl_run = {
                **run_manifest,
                "target_results": target_results,
                "compare_df": compare_df,
                "summary_df": summary_df,
                "best_model_metrics_df": best_metrics_df,
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
            paginated_dataframe(compare_df.round(4), key="train_compare", height=350)
            st.markdown("### Mejor modelo por target según holdout real")
            paginated_dataframe(summary_df, key="train_summary", height=300, hide_index=True)
            st.markdown("### Mejor modelo + métricas (holdout real)")
            if best_metrics_df.empty:
                st.info("No hay métricas detalladas para el mejor modelo.")
            else:
                paginated_dataframe(best_metrics_df.round(4), key="train_best_metrics", height=350, hide_index=True)

        elif st.session_state.automl_run:
            run = st.session_state.automl_run
            st.success(f"✅ Corrida AutoML disponible: `{run['run_id']}`")
            st.caption(f"Artefactos: `{run['base_path']}`")
            st.markdown("### Tabla final target × tipo de modelo")
            paginated_dataframe(run["compare_df"].round(4), key="train_compare_loaded", height=350)
            st.markdown("### Mejor modelo por target según holdout real")
            paginated_dataframe(run["summary_df"], key="train_summary_loaded", height=300, hide_index=True)
            best_metrics_df = run.get("best_model_metrics_df")
            if best_metrics_df is not None and not best_metrics_df.empty:
                st.markdown("### Mejor modelo + métricas (holdout real)")
                paginated_dataframe(best_metrics_df.round(4), key="train_best_loaded", height=350, hide_index=True)
            st.info("Ve a Compare / Evaluate / Explainability para más análisis.")

    elif st.session_state.target_cols:
        targets = st.session_state.target_cols
        target_configs = st.session_state.target_configs or {
            target: infer_target_task(df[target]) for target in targets
        }
        st.info("Targets ya configurados. Entrena o reconfigura arriba.")

        st.divider()
        st.markdown("### ⚙️ Opciones de entrenamiento")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_time_limit = st.slider("Tiempo por target", 30, 1800, 180, 30, format="%d s", key="aml_time2")
        with col2:
            test_size = st.slider("Holdout test size", 0.1, 0.4, 0.2, 0.05, key="aml_test2")
        with col3:
            random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1, key="aml_rs2")
        with col4:
            mode = st.selectbox("Modo mljar", ["Perform", "Explain", "Compete"], index=0, key="aml_mode2")

        if st.button("🚀 Entrenar AutoML", use_container_width=True, type="primary"):
            st.session_state.target_configs = target_configs
            run_id, run_path = create_run_dir()
            quality = analyze_data_quality(df, targets)
            save_json(run_path / "quality_report.json", quality)

            progress = st.progress(0, text="Iniciando...")
            target_results = {}
            errors = {}

            for idx, target in enumerate(targets, start=1):
                progress.progress((idx - 1) / len(targets), text=f"Target `{target}`...")
                try:
                    target_results[target] = run_target_automl(
                        df, target, target_configs[target], all_targets=targets,
                        run_path=run_path, test_size=float(test_size),
                        total_time_limit=int(total_time_limit), mode=mode,
                        random_state=int(random_state),
                    )
                except Exception as exc:
                    errors[target] = str(exc)

            progress.progress(1.0, text="Finalizando...")
            if not target_results:
                st.error("No se pudo entrenar ningún target.")
                st.json(errors)
                st.stop()

            compare_df = build_final_matrix(target_results)
            summary_df = build_target_summary(target_results)
            best_metrics_df = build_best_model_metrics(target_results)
            save_dataframe(run_path / "final_matrix.csv", compare_df.reset_index())
            save_dataframe(run_path / "target_summary.csv", summary_df)

            st.session_state.automl_run = {
                "run_id": run_id, "base_path": str(run_path),
                "target_results": target_results, "compare_df": compare_df,
                "summary_df": summary_df, "best_model_metrics_df": best_metrics_df,
                "errors": errors,
            }
            st.session_state.trained_models = target_results
            st.session_state.compare_df = compare_df
            st.success(f"✅ Corrida `{run_id}` completada.")
            paginated_dataframe(compare_df.round(4), key="train_compare_quick", height=350)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Deep Learning
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dl:
    try:
        import torch
        from ml_pipeline.deep_learning import (
            DL_MODEL_TYPES, DL_DEFAULT_CONFIGS, run_dl_model,
            optimize_dl_hyperparameters,
        )
        torch_available = True
    except ImportError:
        torch_available = False
        st.error("⚠️ PyTorch no está instalado. Ejecuta: `pip install torch`")

    if torch_available:
        st.caption("Modelos de deep learning: autoencoders, VAE, LSTM y GRU con optimización de hiperparámetros.")

        st.divider()
        st.markdown("### 🧩 Selección de modelo")

        dl_model_type = st.selectbox(
            "Tipo de modelo",
            DL_MODEL_TYPES,
            format_func=lambda x: {
                "autoencoder": "Autoencoder (reconstrucción / reducción dimensional)",
                "vae": "Variational Autoencoder (generativo)",
                "lstm": "LSTM (secuencial / serie temporal)",
                "gru": "GRU (secuencial / serie temporal)",
            }.get(x, x),
            key="dl_model_type",
        )

        is_unsupervised = dl_model_type in ("autoencoder", "vae")

        dl_target = None
        if not is_unsupervised:
            dl_target = st.selectbox(
                "Target (variable a predecir)",
                options=df.columns.tolist(),
                key="dl_target",
            )

        st.divider()
        st.markdown("### ⚙️ Hiperparámetros")

        defaults = DL_DEFAULT_CONFIGS[dl_model_type]

        # Apply HPO best config to widget session state BEFORE widgets are rendered
        hpo_applied = st.session_state.get("dl_hpo_apply")
        if hpo_applied and hpo_applied.get("model_type") == dl_model_type:
            best = hpo_applied.get("config", {})
            if dl_model_type in ("autoencoder", "vae"):
                if "encoding_dims" in best:
                    st.session_state.dl_enc_dims = best["encoding_dims"] if isinstance(best["encoding_dims"], str) else ",".join(map(str, best["encoding_dims"]))
                if "learning_rate" in best:
                    st.session_state.dl_lr_ae = float(best["learning_rate"])
                if "batch_size" in best:
                    st.session_state.dl_bs_ae = int(best["batch_size"])
                if "epochs" in best:
                    st.session_state.dl_ep_ae = int(best["epochs"])
                if "dropout" in best:
                    st.session_state.dl_do_ae = float(best["dropout"])
                if "patience" in best:
                    st.session_state.dl_pat_ae = int(best["patience"])
                if "kl_weight" in best:
                    st.session_state.dl_kl = float(best["kl_weight"])
            else:
                if "hidden_dim" in best:
                    st.session_state.dl_hd = int(best["hidden_dim"])
                if "num_layers" in best:
                    st.session_state.dl_nl = int(best["num_layers"])
                if "seq_length" in best:
                    st.session_state.dl_sl = int(best["seq_length"])
                if "learning_rate" in best:
                    st.session_state.dl_lr_seq = float(best["learning_rate"])
                if "batch_size" in best:
                    st.session_state.dl_bs_seq = int(best["batch_size"])
                if "epochs" in best:
                    st.session_state.dl_ep_seq = int(best["epochs"])
                if "dropout" in best:
                    st.session_state.dl_do_seq = float(best["dropout"])
                if "patience" in best:
                    st.session_state.dl_pat_seq = int(best["patience"])
                if "bidirectional" in best:
                    st.session_state.dl_bi = bool(best["bidirectional"])
            # Clear after applying so it doesn't interfere with future runs
            del st.session_state.dl_hpo_apply

        if dl_model_type in ("autoencoder", "vae"):
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                enc_dims = st.text_input(
                    "Encoding dims (comma-separated)",
                    value=",".join(map(str, defaults["encoding_dims"])),
                    key="dl_enc_dims",
                )
            with col2:
                lr = st.number_input("Learning rate", value=float(defaults["learning_rate"]), step=0.0001, format="%.4f", key="dl_lr_ae")
            with col3:
                batch_size = st.number_input("Batch size", value=int(defaults["batch_size"]), step=8, key="dl_bs_ae")
            with col4:
                epochs = st.number_input("Epochs", value=int(defaults["epochs"]), step=10, key="dl_ep_ae")
            with col5:
                dropout = st.number_input("Dropout", value=float(defaults["dropout"]), step=0.05, min_value=0.0, max_value=0.9, format="%.2f", key="dl_do_ae")

            extra_col1, extra_col2 = st.columns(2)
            with extra_col1:
                patience = st.number_input("Early stopping patience", value=int(defaults["patience"]), step=1, key="dl_pat_ae")
            with extra_col2:
                if dl_model_type == "vae":
                    kl_weight = st.number_input("KL weight", value=float(defaults["kl_weight"]), step=0.1, format="%.1f", key="dl_kl")
                else:
                    kl_weight = 1.0

            dl_config = {
                "encoding_dims": [int(x.strip()) for x in enc_dims.split(",") if x.strip()],
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "dropout": dropout,
                "patience": patience,
            }
            if dl_model_type == "vae":
                dl_config["kl_weight"] = kl_weight

        else:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                hidden_dim = st.number_input("Hidden dim", value=int(defaults["hidden_dim"]), step=16, key="dl_hd")
            with col2:
                num_layers = st.number_input("Num layers", value=int(defaults["num_layers"]), min_value=1, max_value=8, step=1, key="dl_nl")
            with col3:
                seq_length = st.number_input("Sequence length", value=int(defaults["seq_length"]), min_value=2, step=1, key="dl_sl")
            with col4:
                lr = st.number_input("Learning rate", value=float(defaults["learning_rate"]), step=0.0001, format="%.4f", key="dl_lr_seq")
            with col5:
                batch_size = st.number_input("Batch size", value=int(defaults["batch_size"]), step=8, key="dl_bs_seq")

            col6, col7, col8, col9 = st.columns(4)
            with col6:
                epochs = st.number_input("Epochs", value=int(defaults["epochs"]), step=10, key="dl_ep_seq")
            with col7:
                dropout = st.number_input("Dropout", value=float(defaults["dropout"]), step=0.05, min_value=0.0, max_value=0.9, format="%.2f", key="dl_do_seq")
            with col8:
                patience = st.number_input("Early stopping patience", value=int(defaults["patience"]), step=1, key="dl_pat_seq")
            with col9:
                bidirectional = st.checkbox("Bidirectional", value=bool(defaults.get("bidirectional", False)), key="dl_bi")

            dl_config = {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "seq_length": seq_length,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "dropout": dropout,
                "patience": patience,
                "bidirectional": bidirectional,
            }

        st.divider()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            dl_test_size = st.slider("Validation size", 0.1, 0.4, 0.2, 0.05, key="dl_test")
        with col_b:
            dl_random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1, key="dl_rs")
        with col_c:
            device_info = "🟢 GPU" if torch.cuda.is_available() else "🔵 CPU"
            st.caption(f"Device: {device_info}")

        st.divider()

        if "dl_hpo_result" not in st.session_state:
            st.session_state.dl_hpo_result = None
        if "dl_results" not in st.session_state:
            st.session_state.dl_results = []
        if "dl_hpo_apply" not in st.session_state:
            st.session_state.dl_hpo_apply = None
        if "dl_prev_model" not in st.session_state:
            st.session_state.dl_prev_model = None
        if "dl_action" not in st.session_state:
            st.session_state.dl_action = None

        if st.session_state.dl_prev_model != dl_model_type:
            st.session_state.dl_prev_model = dl_model_type
            st.session_state.dl_hpo_result = None
            st.session_state.dl_hpo_apply = None

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button(f"🚀 Entrenar {dl_model_type.upper()}", use_container_width=True, type="primary", key="dl_train_btn"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                epoch_info = st.empty()

                def progress_cb(epoch, total, train_loss, val_loss):
                    progress_bar.progress(epoch / total, text=f"Epoch {epoch}/{total}")
                    epoch_info.caption(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

                try:
                    result = run_dl_model(
                        df, model_type=dl_model_type, target=dl_target,
                        config=dl_config, test_size=float(dl_test_size),
                        random_state=int(dl_random_state), progress_callback=progress_cb,
                    )

                    progress_bar.progress(1.0, text="Completado!")
                    status_text.success(f"✅ {dl_model_type.upper()} entrenado exitosamente.")

                    st.session_state.dl_results.append({
                        "model_type": dl_model_type,
                        "target": dl_target,
                        "config": dl_config,
                        "train_rmse": result["train_rmse"],
                        "val_rmse": result["val_rmse"],
                        "history": result["history"],
                        "hpo": False,
                        "model": result.get("model"),
                        "scaler": result.get("scaler"),
                        "feature_cols": result.get("feature_cols", []),
                        "train_reconstructed": result.get("train_reconstructed"),
                        "val_reconstructed": result.get("val_reconstructed"),
                        "train_encoded": result.get("train_encoded"),
                        "val_predictions": result.get("val_predictions"),
                        "reconstruction_error": result.get("reconstruction_error"),
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al entrenar: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        with col_btn2:
            if st.button(f"🔍 Optimizar hiperparámetros ({dl_model_type.upper()})", use_container_width=True, key="dl_hpo_btn"):
                st.session_state.dl_action = "hpo"
                st.rerun()

        if st.session_state.dl_action == "hpo":
            st.markdown(f"### 🔍 Optimización de hiperparámetros — {dl_model_type.upper()}")
            st.caption("Búsqueda exhaustiva (grid search) sobre combinaciones de hiperparámetros.")
            hpo_epochs = st.number_input("Epochs por trial", value=50, min_value=10, step=10, key="hpo_ep")
            hpo_max = st.slider("Máximo de combinaciones a probar", value=30, min_value=5, max_value=200, step=5, key="hpo_n")

            if st.button("▶️ Iniciar búsqueda", type="primary", key="hpo_start"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                trial_info = st.empty()

                def hpo_progress(trial, total, epoch, epoch_total, train_loss, val_loss):
                    overall = (trial + epoch / epoch_total) / total
                    progress_bar.progress(overall, text=f"Trial {trial+1}/{total}, Epoch {epoch}/{epoch_total}")
                    trial_info.caption(f"Trial {trial+1} | Epoch {epoch}/{epoch_total} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

                try:
                    hpo_result = optimize_dl_hyperparameters(
                        df, model_type=dl_model_type, target=dl_target,
                        base_config=dl_config, n_trials=hpo_max,
                        epochs_per_trial=hpo_epochs, test_size=float(dl_test_size),
                        random_state=int(dl_random_state), progress_callback=hpo_progress,
                    )

                    st.session_state.dl_hpo_result = hpo_result

                    progress_bar.progress(1.0, text="Búsqueda completada!")
                    status_text.success(f"✅ Mejor val_loss: {hpo_result['best_val_loss']:.6f}")
                except Exception as e:
                    st.error(f"Error en HPO: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if st.session_state.dl_hpo_result:
            hpo_result = st.session_state.dl_hpo_result

            st.divider()
            st.markdown("### 📊 Resultados de la búsqueda")

            hpo_trials_df = []
            for t in hpo_result["trials"]:
                row = {"Trial": t["trial"], "Val Loss": f"{t['val_loss']:.6f}",
                       "Train RMSE": f"{t['train_rmse']:.4f}", "Val RMSE": f"{t['val_rmse']:.4f}"}
                if t.get("error"):
                    row["Status"] = f"❌ {t.get('error_msg', 'Error')[:60]}"
                else:
                    row["Status"] = "✅"
                hpo_trials_df.append(row)
            paginated_dataframe(pd.DataFrame(hpo_trials_df), key="train_hpo_trials", height=350, hide_index=True)

            st.markdown("### 🏆 Mejor configuración encontrada")
            st.json(hpo_result["best_config"])

            import plotly.graph_objects as go
            fig_hpo = go.Figure()
            valid_trials = [t for t in hpo_result["trials"] if not t.get("error")]
            if valid_trials:
                fig_hpo.add_trace(go.Scatter(
                    x=[t["trial"] for t in valid_trials],
                    y=[t["val_loss"] for t in valid_trials],
                    mode="lines+markers",
                    line=dict(color="#2dd4bf"),
                    marker=dict(size=8),
                    name="Val loss",
                ))
                fig_hpo.add_trace(go.Scatter(
                    x=[t["trial"] for t in valid_trials],
                    y=[t["train_rmse"] for t in valid_trials],
                    mode="lines+markers",
                    line=dict(color="#5b6af0"),
                    marker=dict(size=8),
                    name="Train RMSE",
                ))
            fig_hpo.update_layout(
                title="HPO: Val loss y Train RMSE por trial",
                xaxis_title="Trial",
                paper_bgcolor="#0d0f14",
                plot_bgcolor="#141720",
                font={"color": "#e2e8f0"},
                height=350,
            )
            st.plotly_chart(fig_hpo, use_container_width=True)

            if st.button("🚀 Entrenar con la mejor configuración", type="primary", key="hpo_use_best"):
                best = hpo_result["best_config"].copy()

                st.session_state.dl_hpo_apply = {
                    "model_type": dl_model_type,
                    "config": best,
                }
                st.success("✅ Hiperparámetros rellenados con la mejor configuración. Presiona **Entrenar** para iniciar.")
                st.rerun()

            st.caption("💡 La búsqueda explora combinaciones aleatorias de hiperparámetros y conserva la mejor según validation loss.")

        if st.session_state.get("dl_results"):
            st.divider()
            st.markdown("### 📜 Historial de modelos DL")
            dl_hist = []
            for i, r in enumerate(st.session_state.dl_results):
                dl_hist.append({
                    "#": i + 1,
                    "Modelo": r["model_type"],
                    "Target": r["target"] or "(unsupervised)",
                    "Train RMSE": f"{r['train_rmse']:.4f}",
                    "Val RMSE": f"{r['val_rmse']:.4f}",
                    "Epochs": len(r["history"]["train_loss"]),
                    "HPO": "✅" if r.get("hpo") else "❌",
                })
            paginated_dataframe(pd.DataFrame(dl_hist), key="train_dl_history", height=350, hide_index=True)
