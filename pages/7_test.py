import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from utils.pagination import paginated_dataframe

DARK = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#141720",
    font={"color": "#e2e8f0"},
)

st.markdown("# 🧪 Probar Modelos")
st.markdown('<div class="section-header">Pruebas independientes y encadenamiento de modelos</div>', unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("⚠️ Carga un dataset primero.")
    st.stop()

df = st.session_state.get("df").copy()

saved_models = {}
saved_models.update(st.session_state.get("saved_dl_models", {}))
saved_models.update(st.session_state.get("saved_classic_models", {}))
saved_models.update(st.session_state.get("saved_automl_models", {}))

if not saved_models:
    st.warning("⚠️ No hay modelos guardados. Entrena y guarda modelos en **Visual Train** primero.")
    st.stop()

# ── Initialize test runs ─────────────────────────────────────────────────────
if "test_runs" not in st.session_state:
    st.session_state.test_runs = []

# ── Add/Remove runs ──────────────────────────────────────────────────────────
st.markdown("### ➕ Agregar Prueba")

col_add1, col_add2 = st.columns([3, 1])
with col_add1:
    new_model = st.selectbox("Modelo", list(saved_models.keys()), key="test_add_model")
with col_add2:
    if st.button("➕ Agregar", use_container_width=True, type="primary", key="test_add_btn"):
        st.session_state.test_runs.append({
            "model_name": new_model,
            "source": "dataset",
            "row_idx": 0,
            "manual_values": {},
            "chained_outputs": {},
            "result": None,
        })
        st.rerun()

if not st.session_state.test_runs:
    st.info("Agrega al menos una prueba para comenzar.")
    st.stop()

# ── Render each test run ─────────────────────────────────────────────────────
st.divider()
st.markdown("### 🧪 Pruebas Configuradas")

# Collect available outputs from previous runs for chaining
available_outputs = {}
for i, run in enumerate(st.session_state.test_runs):
    if run.get("result") and run["result"].get("output_values"):
        available_outputs[f"Prueba {i+1} ({run['model_name']})"] = run["result"]["output_values"]

for idx, run in enumerate(st.session_state.test_runs):
    with st.expander(f"🔹 Prueba {idx+1}: {run['model_name']}", expanded=True):
        m_name = run["model_name"]
        m_info = saved_models[m_name]
        model_type = m_info["type"]
        feature_cols = m_info["feature_cols"]
        metadata = m_info["metadata"]
        
        # Model selector
        col_m1, col_m2 = st.columns([3, 1])
        with col_m1:
            new_model = st.selectbox("Modelo", list(saved_models.keys()), index=list(saved_models.keys()).index(m_name), key=f"test_model_{idx}")
            if new_model != m_name:
                st.session_state.test_runs[idx]["model_name"] = new_model
                st.session_state.test_runs[idx]["result"] = None
                m_name = new_model
                m_info = saved_models[m_name]
                model_type = m_info["type"]
                feature_cols = m_info["feature_cols"]
                metadata = m_info["metadata"]
        
        with col_m2:
            if st.button("🗑️", key=f"test_remove_{idx}"):
                st.session_state.test_runs.pop(idx)
                st.rerun()
        
        # Determine which columns need input
        is_sequence = model_type in ("lstm", "gru")
        is_classic = model_type == "classic"
        if is_classic:
            input_cols = m_info.get("feature_cols", feature_cols)
        elif is_sequence:
            input_cols = metadata.get("input_cols", feature_cols)
        else:
            input_cols = feature_cols
        
        # Input source
        source_options = ["Seleccionar fila del dataset"]
        if available_outputs:
            source_options.append("Usar output de otra prueba")
        source_options.append("Ingresar manualmente")
        
        source = st.radio("Fuente de entrada", source_options, horizontal=True, key=f"test_source_{idx}")
        st.session_state.test_runs[idx]["source"] = source
        
        input_row = None
        
        if source == "Seleccionar fila del dataset":
            max_rows = min(1000, len(df))
            row_idx = st.slider("Índice de fila", 0, max_rows - 1, run.get("row_idx", 0), key=f"test_row_{idx}")
            st.session_state.test_runs[idx]["row_idx"] = row_idx
            raw_row = df.iloc[row_idx:row_idx+1][input_cols]
            st.markdown("#### Datos de entrada (editables)")
            input_row = st.data_editor(raw_row, num_rows="fixed", key=f"test_editor_{idx}", use_container_width=True)
            
        elif source == "Usar output de otra prueba":
            chain_source = st.selectbox("Prueba origen", list(available_outputs.keys()), key=f"test_chain_{idx}")
            output_vals = available_outputs[chain_source]
            
            st.caption("Mapea los outputs de la prueba origen a las columnas de input de este modelo:")
            manual_vals = {}
            for fc in input_cols:
                if fc in output_vals:
                    manual_vals[fc] = output_vals[fc]
                else:
                    series = df[fc]
                    if pd.api.types.is_numeric_dtype(series):
                        clean = series.dropna()
                        val = float(clean.median()) if not clean.empty else 0.0
                        manual_vals[fc] = val
                    else:
                        manual_vals[fc] = ""
            
            raw_row = pd.DataFrame([manual_vals])
            st.markdown("#### Datos de entrada (editables)")
            input_row = st.data_editor(raw_row, num_rows="fixed", key=f"test_editor_{idx}", use_container_width=True)
        
        else:
            st.caption(f"Completa las **{len(input_cols)} columnas de entrada** requeridas:")
            manual_vals = {}
            for fc in input_cols:
                series = df[fc]
                if pd.api.types.is_numeric_dtype(series):
                    clean = series.dropna()
                    val = float(clean.median()) if not clean.empty else 0.0
                    manual_vals[fc] = val
                else:
                    manual_vals[fc] = ""
            raw_row = pd.DataFrame([manual_vals])
            st.markdown("#### Datos de entrada (editables)")
            input_row = st.data_editor(raw_row, num_rows="fixed", key=f"test_editor_{idx}", use_container_width=True)
        
        # Run button for this test
        if st.button("▶️ Ejecutar", type="primary", key=f"test_run_{idx}"):
            try:
                result = {"model": m_name, "type": model_type}
                
                if model_type == "automl":
                    feature_cols = m_info["feature_cols"]
                    automl_obj = m_info["automl"]
                    task = m_info.get("task", "classification")
                    
                    row_data = input_row[feature_cols]
                    pred = automl_obj.predict(row_data)[0]
                    
                    if task == "classification":
                        result["prediction"] = str(pred)
                        result["prediction_value"] = str(pred)
                        try:
                            proba = automl_obj.predict_proba(row_data)[0].tolist()
                            result["probabilities"] = proba
                            result["output_values"] = {"prediction": str(pred)}
                            for i, p in enumerate(proba):
                                result["output_values"][f"prob_class_{i}"] = float(p)
                        except Exception:
                            result["output_values"] = {"prediction": str(pred)}
                    else:
                        result["prediction"] = float(pred)
                        result["output_values"] = {"prediction": float(pred)}
                    
                elif model_type == "classic":
                    feature_cols = m_info["feature_cols"]
                    scaler = m_info["scaler"]
                    le = m_info.get("label_encoder")
                    task = m_info.get("task", "classification")
                    
                    row_data = input_row[feature_cols].values.astype(float)
                    row_data = np.nan_to_num(row_data, nan=0.0)
                    row_scaled = scaler.transform(row_data)
                    
                    model = m_info["model"]
                    pred = model.predict(row_scaled)[0]
                    
                    if le:
                        pred_label = str(le.inverse_transform([int(pred)])[0])
                        proba = None
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(row_scaled)[0].tolist()
                        result["prediction"] = pred_label
                        result["prediction_value"] = float(pred)
                        result["probabilities"] = proba
                        result["output_values"] = {"prediction": pred_label}
                        if proba:
                            for i, p in enumerate(proba):
                                result["output_values"][f"prob_class_{i}"] = float(p)
                    else:
                        result["prediction"] = float(pred)
                        result["output_values"] = {"prediction": float(pred)}
                    
                elif model_type in ("autoencoder", "vae"):
                    scaler = m_info["scaler"]
                    if scaler:
                        row_scaled = scaler.transform(row_data)
                    else:
                        row_scaled = row_data
                    
                    x = torch.tensor(row_scaled, dtype=torch.float32)
                    with torch.no_grad():
                        output = m_info["model"](x)
                        recon = output[0].numpy() if isinstance(output, tuple) else output.numpy()
                    
                    error = float(np.mean((row_scaled - recon) ** 2))
                    train_errors = metadata.get("reconstruction_error")
                    anomaly_status = "🟢 Normal"
                    if train_errors is not None:
                        p95 = float(np.percentile(train_errors, 95))
                        p99 = float(np.percentile(train_errors, 99))
                        if error > p99:
                            anomaly_status = "🔴 Anómalo"
                        elif error > p95:
                            anomaly_status = "🟡 Sospechoso"
                    
                    result["error"] = error
                    result["anomaly"] = anomaly_status
                    result["reconstructed"] = recon.tolist()
                    result["input_scaled"] = row_scaled.tolist()
                    result["output_values"] = {f"recon_{c}": float(recon[0, i]) for i, c in enumerate(input_cols)}
                    
                else:
                    input_scaler = m_info.get("input_scaler")
                    if input_scaler:
                        input_scaled = input_scaler.transform(row_data).flatten()
                    else:
                        input_scaled = row_data.flatten()
                    
                    predict_cols = metadata.get("predict_cols", [])
                    x = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        preds = m_info["model"](x, teacher_forcing_prob=0.0).numpy().flatten()
                    
                    result["predictions"] = {c: float(v) for c, v in zip(predict_cols, preds)}
                    result["output_values"] = result["predictions"]
                
                st.session_state.test_runs[idx]["result"] = result
                
            except Exception as e:
                st.session_state.test_runs[idx]["result"] = {"error": str(e)}
            
            st.rerun()
        
        # Show result
        if run.get("result"):
            res = run["result"]
            if "error" in res and "type" not in res:
                st.error(f"Error: {res['error']}")
            else:
                st.success("✅ Ejecutado")
                
                if res["type"] == "automl":
                    st.metric("Predicción", str(res.get("prediction", "-")))
                    if res.get("probabilities"):
                        task = m_info.get("task", "classification")
                        classes = list(df[m_info["target_col"]].dropna().unique())
                        prob_df = pd.DataFrame([{"Clase": str(c), "Probabilidad": p} for c, p in zip(classes, res["probabilities"])])
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=prob_df["Clase"], y=prob_df["Probabilidad"], marker_color="#2dd4bf", text=prob_df["Probabilidad"].round(3), textposition="outside"))
                        fig.update_layout(title="Probabilidades por clase", **DARK, height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif res["type"] == "classic":
                    st.metric("Predicción", str(res.get("prediction", "-")))
                    if res.get("probabilities"):
                        classes = list(df[m_info["target_col"]].dropna().unique())
                        prob_df = pd.DataFrame([{"Clase": str(c), "Probabilidad": p} for c, p in zip(classes, res["probabilities"])])
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=prob_df["Clase"], y=prob_df["Probabilidad"], marker_color="#2dd4bf", text=prob_df["Probabilidad"].round(3), textposition="outside"))
                        fig.update_layout(title="Probabilidades por clase", **DARK, height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif res["type"] in ("autoencoder", "vae"):
                    st.metric("Error de reconstrucción", f"{res['error']:.6f}")
                    st.metric("Estado", res["anomaly"])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=input_cols, y=np.array(res["input_scaled"]).flatten(), name="Original", marker_color="#5b6af0"))
                    fig.add_trace(go.Bar(x=input_cols, y=np.array(res["reconstructed"]).flatten(), name="Reconstruido", marker_color="#2dd4bf"))
                    fig.update_layout(title="Original vs Reconstruido", barmode="group", **DARK, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    pred_df = pd.DataFrame([res["predictions"]])
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=list(res["predictions"].keys()), y=list(res["predictions"].values()), marker_color="#2dd4bf"))
                    fig.update_layout(title="Predicciones", **DARK, height=300)
                    st.plotly_chart(fig, use_container_width=True)

# ── Comparison ────────────────────────────────────────────────────────────────
completed = [r for r in st.session_state.test_runs if r.get("result") and "type" in r["result"]]
if len(completed) > 1:
    st.divider()
    st.markdown("### 📊 Comparación")
    
    comp_data = []
    for r in completed:
        res = r["result"]
        row = {"Prueba": r["model_name"], "Tipo": res["type"].upper()}
        if res["type"] == "automl":
            row["Error"] = "-"
            row["Estado"] = "-"
            row["Predicción"] = str(res.get("prediction", "-"))
        elif res["type"] == "classic":
            row["Error"] = "-"
            row["Estado"] = "-"
            row["Predicción"] = str(res.get("prediction", "-"))
        elif res["type"] in ("autoencoder", "vae"):
            row["Error"] = res.get("error", "-")
            row["Estado"] = res.get("anomaly", "-")
            row["Predicción"] = "-"
        else:
            row["Error"] = "-"
            row["Estado"] = "-"
            row["Predicción"] = ", ".join([f"{k}: {v:.3f}" for k, v in res.get("predictions", {}).items()])
        comp_data.append(row)
    
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    
    unsup = [r for r in completed if r["result"]["type"] in ("autoencoder", "vae")]
    if unsup:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[r["model_name"] for r in unsup],
            y=[r["result"]["error"] for r in unsup],
            marker_color="#5b6af0",
            text=[f"{r['result']['error']:.4f}" for r in unsup],
            textposition="outside",
        ))
        fig.update_layout(title="Error de Reconstrucción", **DARK, height=350)
        st.plotly_chart(fig, use_container_width=True)
