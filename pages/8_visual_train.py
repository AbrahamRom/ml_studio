import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import warnings

from ml_pipeline.deep_learning import DL_DEFAULT_CONFIGS, run_dl_model

warnings.filterwarnings("ignore")

st.markdown("# 🎨 Entrenamiento Visual")
st.markdown('<div class="section-header">Configuración interactiva de modelos Deep Learning</div>', unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("⚠️ Carga un dataset primero.")
    st.stop()

df = st.session_state.get("df").copy()

# ── Step 1: Model selection ───────────────────────────────────────────────────
st.markdown("### 1️⃣ Seleccionar modelo")

model_type = st.selectbox(
    "Tipo de modelo",
    ["autoencoder", "vae", "lstm", "gru"],
    format_func=lambda x: {
        "autoencoder": "Autoencoder (reconstrucción / detección de anomalías)",
        "vae": "VAE - Variational Autoencoder (generativo)",
        "lstm": "LSTM (cadena secuencial autoregresiva)",
        "gru": "GRU (cadena secuencial autoregresiva)",
    }.get(x, x),
    key="visual_model_type",
)

is_unsupervised = model_type in ("autoencoder", "vae")
is_sequence = model_type in ("lstm", "gru")

# ── Step 2: Column configuration ──────────────────────────────────────────────
st.markdown("### 2️⃣ Configurar columnas")

if is_unsupervised:
    st.caption(f"Selecciona las columnas para entrenar el {model_type.upper()}.")
    col_options = df.columns.tolist()
    selected_cols = st.multiselect(
        "Columnas de entrada",
        options=col_options,
        default=col_options[:min(6, len(col_options))],
        key="visual_unsup_cols",
    )
    if not selected_cols:
        st.warning("Selecciona al menos una columna.")
        st.stop()
    feature_cols = selected_cols
    target_col = None

else:
    st.caption("Define la cadena secuencial. El modelo predecirá de forma **autoregresiva**: de las Inputs deduce la 1ra Predecir, de Inputs + 1ra deduce la 2da, etc.")

    col_options = ["(vacío)"] + df.columns.tolist()

    if "visual_slot_count" not in st.session_state:
        st.session_state.visual_slot_count = 3

    slot_count = st.slider("Número de ranuras", 1, len(df.columns), st.session_state.visual_slot_count, key="visual_slot_slider")
    st.session_state.visual_slot_count = slot_count

    slots = []
    for i in range(slot_count):
        st.markdown(f"**Ranura {i + 1}**")
        c1, c2 = st.columns([2, 1])
        with c1:
            col_sel = st.selectbox(
                "Columna",
                options=col_options,
                index=0,
                label_visibility="collapsed",
                key=f"visual_slot_col_{i}",
            )
        with c2:
            if col_sel != "(vacío)":
                role = st.radio(
                    "Rol",
                    ["Input", "Predecir"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"visual_slot_role_{i}",
                )
            else:
                role = None
        if col_sel != "(vacío)":
            slots.append({"index": i, "column": col_sel, "role": role})

    if not slots:
        st.warning("Configura al menos una ranura con una columna.")
        st.stop()

    input_slots = [s for s in slots if s["role"] == "Input"]
    predict_slots = [s for s in slots if s["role"] == "Predecir"]

    if not input_slots:
        st.warning("Necesitas al menos una ranura marcada como **Input**.")
        st.stop()
    if not predict_slots:
        st.warning("Necesitas al menos una ranura marcada como **Predecir**.")
        st.stop()

    st.divider()
    st.markdown("#### Cadena de predicción autoregresiva")
    chain_steps = []
    current_inputs = [s["column"] for s in input_slots]
    for s in predict_slots:
        step_desc = f"🔮 Predecir `{s['column']}` usando: `[{', '.join(current_inputs)}]`"
        chain_steps.append(step_desc)
        current_inputs.append(s["column"])

    for step in chain_steps:
        st.caption(step)

    feature_cols = [s["column"] for s in slots]
    target_col = None

# ── Step 3: Hyperparameters State Management ──────────────────────────────────
st.markdown("### ⚙️ Hiperparámetros")

defaults = DL_DEFAULT_CONFIGS[model_type]

def init_visual_state():
    st.session_state.visual_enc_dims = ",".join(map(str, defaults.get("encoding_dims", [64, 32])))
    st.session_state.visual_lr_unsup = float(defaults.get("learning_rate", 0.001))
    st.session_state.visual_bs_unsup = int(defaults.get("batch_size", 32))
    st.session_state.visual_ep_unsup = int(defaults.get("epochs", 100))
    st.session_state.visual_do_unsup = float(defaults.get("dropout", 0.0))
    st.session_state.visual_pat_unsup = int(defaults.get("patience", 10))
    st.session_state.visual_kl = float(defaults.get("kl_weight", 1.0))
    st.session_state.visual_hd = int(defaults.get("hidden_dim", 64))
    st.session_state.visual_nl = int(defaults.get("num_layers", 2))
    st.session_state.visual_sl = int(defaults.get("seq_length", 5))
    st.session_state.visual_lr_seq = float(defaults.get("learning_rate", 0.001))
    st.session_state.visual_bs_seq = int(defaults.get("batch_size", 32))
    st.session_state.visual_ep_seq = int(defaults.get("epochs", 100))
    st.session_state.visual_do_seq = float(defaults.get("dropout", 0.1))
    st.session_state.visual_pat_seq = int(defaults.get("patience", 10))
    st.session_state.visual_bi = bool(defaults.get("bidirectional", False))
    st.session_state.visual_prev_model = model_type

if "visual_prev_model" not in st.session_state or st.session_state.visual_prev_model != model_type:
    init_visual_state()

# Apply HPO best config BEFORE rendering widgets
if st.session_state.get("visual_hpo_best_config"):
    best = st.session_state.visual_hpo_best_config
    if is_unsupervised:
        if "encoding_dims" in best:
            st.session_state.visual_enc_dims = ",".join(map(str, best["encoding_dims"]))
        if "learning_rate" in best:
            st.session_state.visual_lr_unsup = float(best["learning_rate"])
        if "batch_size" in best:
            st.session_state.visual_bs_unsup = int(best["batch_size"])
        if "epochs" in best:
            st.session_state.visual_ep_unsup = int(best["epochs"])
        if "dropout" in best:
            st.session_state.visual_do_unsup = float(best["dropout"])
        if "patience" in best:
            st.session_state.visual_pat_unsup = int(best["patience"])
        if "kl_weight" in best:
            st.session_state.visual_kl = float(best["kl_weight"])
    else:
        if "hidden_dim" in best:
            st.session_state.visual_hd = int(best["hidden_dim"])
        if "num_layers" in best:
            st.session_state.visual_nl = int(best["num_layers"])
        if "learning_rate" in best:
            st.session_state.visual_lr_seq = float(best["learning_rate"])
        if "batch_size" in best:
            st.session_state.visual_bs_seq = int(best["batch_size"])
        if "epochs" in best:
            st.session_state.visual_ep_seq = int(best["epochs"])
        if "dropout" in best:
            st.session_state.visual_do_seq = float(best["dropout"])
        if "patience" in best:
            st.session_state.visual_pat_seq = int(best["patience"])
    
    del st.session_state.visual_hpo_best_config
    st.rerun()

if "visual_enc_dims" not in st.session_state:
    init_visual_state()

# ── Step 4: Hyperparameters UI ────────────────────────────────────────────────
if is_unsupervised:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        enc_dims = st.text_input("Encoding dims (comma-separated)", value=st.session_state.visual_enc_dims, key="visual_enc_dims")
    with col2:
        lr = st.number_input("Learning rate", value=st.session_state.visual_lr_unsup, step=0.0001, format="%.4f", key="visual_lr_unsup")
    with col3:
        batch_size = st.number_input("Batch size", value=st.session_state.visual_bs_unsup, step=8, key="visual_bs_unsup")
    with col4:
        epochs = st.number_input("Epochs", value=st.session_state.visual_ep_unsup, step=10, key="visual_ep_unsup")
    with col5:
        dropout = st.number_input("Dropout", value=st.session_state.visual_do_unsup, step=0.05, min_value=0.0, max_value=0.9, format="%.2f", key="visual_do_unsup")
    extra_col1, extra_col2 = st.columns(2)
    with extra_col1:
        patience = st.number_input("Early stopping patience", value=st.session_state.visual_pat_unsup, step=1, key="visual_pat_unsup")
    with extra_col2:
        kl_weight = st.number_input("KL weight", value=st.session_state.visual_kl, step=0.1, format="%.1f", key="visual_kl") if model_type == "vae" else 1.0

    dl_config = {
        "encoding_dims": [int(x.strip()) for x in enc_dims.split(",") if x.strip()],
        "learning_rate": lr, "batch_size": batch_size, "epochs": epochs,
        "dropout": dropout, "patience": patience,
    }
    if model_type == "vae":
        dl_config["kl_weight"] = kl_weight

else:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        hidden_dim = st.number_input("Hidden dim", value=st.session_state.visual_hd, step=16, key="visual_hd")
    with col2:
        num_layers = st.number_input("Num layers", value=st.session_state.visual_nl, min_value=1, max_value=8, step=1, key="visual_nl")
    with col3:
        seq_length = st.number_input("Sequence length (ventana temporal)", value=st.session_state.visual_sl, min_value=2, step=1, key="visual_sl")
    with col4:
        lr = st.number_input("Learning rate", value=st.session_state.visual_lr_seq, step=0.0001, format="%.4f", key="visual_lr_seq")
    with col5:
        batch_size = st.number_input("Batch size", value=st.session_state.visual_bs_seq, step=8, key="visual_bs_seq")
    col6, col7, col8, col9 = st.columns(4)
    with col6:
        epochs = st.number_input("Epochs", value=st.session_state.visual_ep_seq, step=10, key="visual_ep_seq")
    with col7:
        dropout = st.number_input("Dropout", value=st.session_state.visual_do_seq, step=0.05, min_value=0.0, max_value=0.9, format="%.2f", key="visual_do_seq")
    with col8:
        patience = st.number_input("Early stopping patience", value=st.session_state.visual_pat_seq, step=1, key="visual_pat_seq")
    with col9:
        bidirectional = st.checkbox("Bidirectional", value=st.session_state.visual_bi, key="visual_bi")

    dl_config = {
        "hidden_dim": hidden_dim, "num_layers": num_layers, "seq_length": seq_length,
        "learning_rate": lr, "batch_size": batch_size, "epochs": epochs,
        "dropout": dropout, "patience": patience, "bidirectional": bidirectional,
    }

st.divider()
col_a, col_b = st.columns(2)
with col_a:
    test_size = st.slider("Validation size", 0.1, 0.4, 0.2, 0.05, key="visual_test")
with col_b:
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1, key="visual_rs")

# ── Step 5: Train & HPO ───────────────────────────────────────────────────────
st.divider()
st.markdown("### 🚀 Entrenar / Optimizar")

col_btn1, col_btn2 = st.columns(2)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_chain_data(df, slots, seq_length, test_size, random_state):
    all_cols = [s["column"] for s in slots]
    input_cols = [s["column"] for s in slots if s["role"] == "Input"]
    predict_cols = [s["column"] for s in slots if s["role"] == "Predecir"]

    data = df[all_cols].copy().dropna()
    if len(data) < seq_length + len(predict_cols):
        raise ValueError(f"Se necesitan al menos {seq_length + len(predict_cols)} filas sin nulos. Hay {len(data)}.")

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.values)

    input_scaler = StandardScaler()
    input_scaler.fit(df[input_cols].dropna().values)

    col_indices = {col: i for i, col in enumerate(all_cols)}
    input_indices = [col_indices[c] for c in input_cols]
    predict_indices = [col_indices[c] for c in predict_cols]

    X_seq, y_targets = [], []
    for i in range(len(data_scaled) - seq_length - len(predict_cols) + 1):
        window = data_scaled[i:i + seq_length]
        input_features = window[-1, input_indices]
        targets = [data_scaled[i + seq_length + j, p_idx] for j, p_idx in enumerate(predict_indices)]
        X_seq.append(input_features)
        y_targets.append(targets)

    X = np.array(X_seq, dtype=np.float32)
    y = np.array(y_targets, dtype=np.float32)
    return train_test_split(X, y, test_size=test_size, random_state=random_state), scaler, input_scaler, input_cols, predict_cols, all_cols

class AutoregressiveChain(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_predict_steps, dropout=0.0, bidirectional=False, model_type="lstm"):
        super().__init__()
        self.num_predict_steps = num_predict_steps
        if model_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(rnn_out_dim + 1, 1)

    def forward(self, x, teacher_forcing_targets=None, teacher_forcing_prob=0.5):
        batch_size = x.size(0)
        device = x.device
        output, hidden = self.rnn(x, None)
        last_hidden = output[:, -1, :]
        predictions = []
        prev_pred = torch.zeros(batch_size, 1, device=device)
        for step in range(self.num_predict_steps):
            decoder_input = torch.cat([last_hidden, prev_pred], dim=1)
            pred = self.fc(decoder_input)
            predictions.append(pred)
            if self.training and teacher_forcing_targets is not None and np.random.rand() < teacher_forcing_prob:
                prev_pred = teacher_forcing_targets[:, step:step+1]
            else:
                prev_pred = pred
        return torch.cat(predictions, dim=1)

def train_chain_model(X_train, y_train, X_val, y_val, config, model_type, num_predict_steps, progress_callback=None):
    set_seed(config.get("random_state", 42))
    input_dim = X_train.shape[1]
    hidden_dim = config.get("hidden_dim", 64)
    num_layers = config.get("num_layers", 2)
    lr = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 100)
    dropout = config.get("dropout", 0.1)
    patience = config.get("patience", 10)
    bidirectional = config.get("bidirectional", False)

    model = AutoregressiveChain(input_dim, hidden_dim, num_layers, num_predict_steps, dropout, bidirectional, model_type).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=batch_size, shuffle=True)
    X_val_t, y_val_t = torch.FloatTensor(X_val).to(DEVICE), torch.FloatTensor(y_val).to(DEVICE)

    best_val_loss, best_state, patience_counter = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            tf_prob = max(0.0, 1.0 - epoch / epochs)
            output = model(batch_x.unsqueeze(1), batch_y, tf_prob)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t.unsqueeze(1), teacher_forcing_prob=0.0), y_val_t).item()

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience: break

        if progress_callback: progress_callback(epoch, epochs, avg_train_loss, val_loss)

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.FloatTensor(X_train).unsqueeze(1).to(DEVICE), teacher_forcing_prob=0.0).cpu().numpy()
        val_pred = model(X_val_t.unsqueeze(1), teacher_forcing_prob=0.0).cpu().numpy()

    per_step_rmse = [float(np.sqrt(np.mean((y_val[:, i] - val_pred[:, i]) ** 2))) for i in range(y_val.shape[1])]
    return {
        "model": model, "history": history,
        "train_rmse": float(np.sqrt(np.mean((y_train - train_pred) ** 2))),
        "val_rmse": float(np.sqrt(np.mean((y_val - val_pred) ** 2))),
        "train_predictions": train_pred, "val_predictions": val_pred,
        "per_step_rmse": per_step_rmse,
    }

def optimize_chain_hyperparameters(X_train, y_train, X_val, y_val, base_config, model_type, num_predict_steps, n_trials=20, epochs_per_trial=30, progress_callback=None):
    set_seed(base_config.get("random_state", 42))
    search_space = {
        "hidden_dim": [32, 64, 128], "num_layers": [1, 2],
        "learning_rate": [0.01, 0.001, 0.0001], "batch_size": [32, 64], "dropout": [0.0, 0.2],
    }
    trials, best_val_loss, best_config = [], float("inf"), None

    for trial in range(n_trials):
        trial_config = {
            "hidden_dim": random.choice(search_space["hidden_dim"]),
            "num_layers": random.choice(search_space["num_layers"]),
            "learning_rate": random.choice(search_space["learning_rate"]),
            "batch_size": random.choice(search_space["batch_size"]),
            "dropout": random.choice(search_space["dropout"]),
            "epochs": epochs_per_trial, "patience": 5, "random_state": base_config.get("random_state", 42),
        }
        try:
            result = train_chain_model(X_train, y_train, X_val, y_val, trial_config, model_type, num_predict_steps)
            val_loss = result["val_rmse"]
            trials.append({"trial": trial, "val_loss": val_loss, "train_rmse": result["train_rmse"], "config": trial_config, "error": False})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = trial_config.copy()
        except Exception as e:
            trials.append({"trial": trial, "val_loss": float("inf"), "config": trial_config, "error": True, "error_msg": str(e)})
        if progress_callback: progress_callback(trial, n_trials, 0, 1, 0, 0)

    return {"best_config": best_config, "best_val_loss": best_val_loss, "trials": trials}

def optimize_unsupervised_hyperparameters(df, feature_cols, model_type, base_config, n_trials=20, epochs_per_trial=30, test_size=0.2, random_state=42, progress_callback=None):
    search_space = {
        "encoding_dims": [[128, 64], [64, 32], [64, 32, 16], [32, 16], [128, 64, 32, 16]],
        "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [16, 32, 64, 128],
        "dropout": [0.0, 0.1, 0.2, 0.3],
    }
    if model_type == "vae":
        search_space["kl_weight"] = [0.1, 0.5, 1.0, 2.0]

    trials, best_val_loss, best_config = [], float("inf"), None
    for trial in range(n_trials):
        trial_config = {
            "encoding_dims": random.choice(search_space["encoding_dims"]),
            "learning_rate": random.choice(search_space["learning_rate"]),
            "batch_size": random.choice(search_space["batch_size"]),
            "dropout": random.choice(search_space["dropout"]),
            "epochs": epochs_per_trial, "patience": 5, "random_state": random_state,
        }
        if model_type == "vae":
            trial_config["kl_weight"] = random.choice(search_space["kl_weight"])

        try:
            result = run_dl_model(df[feature_cols], model_type=model_type, target=None, config=trial_config, test_size=test_size, random_state=random_state)
            val_loss = result["val_rmse"]
            trials.append({"trial": trial, "val_loss": val_loss, "train_rmse": result["train_rmse"], "config": trial_config, "error": False})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = trial_config.copy()
        except Exception as e:
            trials.append({"trial": trial, "val_loss": float("inf"), "config": trial_config, "error": True, "error_msg": str(e)})
        if progress_callback: progress_callback(trial, n_trials, 0, 1, 0, 0)

    return {"best_config": best_config, "best_val_loss": best_val_loss, "trials": trials}

with col_btn1:
    if st.button("🚀 Entrenar modelo", use_container_width=True, type="primary", key="visual_train_btn"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            epoch_info = st.empty()

            def progress_cb(epoch, total, train_loss, val_loss):
                progress_bar.progress(epoch / total, text=f"Epoch {epoch}/{total}")
                epoch_info.caption(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

            if is_unsupervised:
                result = run_dl_model(df[feature_cols], model_type=model_type, target=None, config=dl_config, test_size=float(test_size), random_state=int(random_state), progress_callback=progress_cb)
            else:
                (X_train, X_val, y_train, y_val), scaler, input_scaler, input_cols, predict_cols, all_cols = prepare_chain_data(df, slots, dl_config["seq_length"], float(test_size), int(random_state))
                result = train_chain_model(X_train, y_train, X_val, y_val, dl_config, model_type, len(predict_cols), progress_cb)
                result.update({"scaler": scaler, "input_scaler": input_scaler, "feature_cols": all_cols, "input_cols": input_cols, "predict_cols": predict_cols, "config": dl_config, "model_type": model_type, "target": None, "train_rows": len(X_train), "val_rows": len(X_val), "slots": slots})

            progress_bar.progress(1.0, text="Completado!")
            status_text.success(f"✅ {model_type.upper()} entrenado exitosamente.")

            if "dl_results" not in st.session_state: st.session_state.dl_results = []
            st.session_state.dl_results.append({
                "model_type": model_type, "target": target_col, "config": dl_config,
                "train_rmse": result["train_rmse"], "val_rmse": result["val_rmse"],
                "history": result["history"], "hpo": False,
                "model": result.get("model"), "scaler": result.get("scaler"),
                "input_scaler": result.get("input_scaler"),
                "feature_cols": result.get("feature_cols", []),
                "train_reconstructed": result.get("train_reconstructed"),
                "val_reconstructed": result.get("val_reconstructed"),
                "train_encoded": result.get("train_encoded"),
                "val_predictions": result.get("val_predictions"),
                "reconstruction_error": result.get("reconstruction_error"),
                "slots": slots if is_sequence else None,
                "per_step_rmse": result.get("per_step_rmse"),
                "predict_cols": result.get("predict_cols"),
                "input_cols": result.get("input_cols"),
            })
            st.rerun()
        except Exception as e:
            st.error(f"Error al entrenar: {e}")
            import traceback
            st.code(traceback.format_exc())

with col_btn2:
    if st.button("🔍 Optimizar hiperparámetros", use_container_width=True, key="visual_hpo_btn"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            trial_info = st.empty()

            def hpo_progress(trial, total, epoch, epoch_total, train_loss, val_loss):
                progress_bar.progress((trial + 1) / total, text=f"Trial {trial+1}/{total}")
                trial_info.caption(f"Completado {trial+1} de {total}")

            if is_sequence:
                (X_train, X_val, y_train, y_val), scaler, input_scaler, input_cols, predict_cols, all_cols = prepare_chain_data(df, slots, dl_config["seq_length"], float(test_size), int(random_state))
                hpo_result = optimize_chain_hyperparameters(X_train, y_train, X_val, y_val, dl_config, model_type, len(predict_cols), n_trials=20, epochs_per_trial=30, progress_callback=hpo_progress)
            else:
                hpo_result = optimize_unsupervised_hyperparameters(df, feature_cols, model_type, dl_config, n_trials=20, epochs_per_trial=30, test_size=float(test_size), random_state=int(random_state), progress_callback=hpo_progress)

            if hpo_result["best_config"]:
                st.session_state.visual_hpo_best_config = hpo_result["best_config"]
                st.success("✅ Hiperparámetros actualizados con la mejor configuración.")
                st.rerun()
            else:
                st.warning("No se encontró una configuración válida.")

            progress_bar.progress(1.0, text="Búsqueda completada!")
            status_text.success(f"✅ Mejor RMSE: {hpo_result['best_val_loss']:.4f}")

            st.divider()
            st.markdown("### 🏆 Mejor configuración")
            if hpo_result["best_config"]:
                st.json(hpo_result["best_config"])

            fig = go.Figure()
            valid_trials = [t for t in hpo_result["trials"] if not t.get("error")]
            if valid_trials:
                fig.add_trace(go.Scatter(x=[t["trial"] for t in valid_trials], y=[t["val_loss"] for t in valid_trials], mode="lines+markers", line=dict(color="#2dd4bf"), marker=dict(size=8), name="Val RMSE"))
            fig.update_layout(title="HPO: Val RMSE por trial", xaxis_title="Trial", yaxis_title="RMSE", paper_bgcolor="#0d0f14", plot_bgcolor="#141720", font={"color": "#e2e8f0"}, height=350)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error en HPO: {e}")
            import traceback
            st.code(traceback.format_exc())

# ── Results visualization ─────────────────────────────────────────────────────
if st.session_state.get("dl_results"):
    last_result = st.session_state.dl_results[-1]
    if last_result["model_type"] == model_type:
        st.divider()
        st.markdown("### 📊 Resultados")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train RMSE", f"{last_result['train_rmse']:.4f}")
        c2.metric("Val RMSE", f"{last_result['val_rmse']:.4f}")
        c3.metric("Gap", f"{abs(last_result['val_rmse'] - last_result['train_rmse']):.4f}")
        c4.metric("Epochs", len(last_result["history"]["train_loss"]))

        if is_sequence and last_result.get("per_step_rmse"):
            st.markdown("#### RMSE por paso de predicción")
            steps = last_result.get("predict_cols", [f"Paso {i+1}" for i in range(len(last_result["per_step_rmse"]))])
            fig_steps = go.Figure(go.Bar(x=steps, y=last_result["per_step_rmse"], marker_color="#5b6af0", text=[f"{v:.4f}" for v in last_result["per_step_rmse"]], textposition="outside"))
            fig_steps.update_layout(title="Error por eslabón de la cadena", xaxis_title="Columna predicha", yaxis_title="RMSE", paper_bgcolor="#0d0f14", plot_bgcolor="#141720", font={"color": "#e2e8f0"}, height=350)
            st.plotly_chart(fig_steps, use_container_width=True)

        tab1, tab2 = st.tabs(["📈 Loss", "🔮 Predicciones vs Reales"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=last_result["history"]["train_loss"], name="Train loss", line=dict(color="#5b6af0")))
            fig.add_trace(go.Scatter(y=last_result["history"]["val_loss"], name="Val loss", line=dict(color="#2dd4bf")))
            fig.update_layout(title="Loss por epoch", xaxis_title="Epoch", yaxis_title="Loss", paper_bgcolor="#0d0f14", plot_bgcolor="#141720", font={"color": "#e2e8f0"}, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if is_sequence and last_result.get("val_predictions") is not None:
                val_pred = last_result["val_predictions"]
                fig_pred = go.Figure()
                for i in range(val_pred.shape[1]):
                    col_name = last_result.get("predict_cols", [f"Paso {i+1}"])[i]
                    fig_pred.add_trace(go.Box(y=val_pred[:, i], name=col_name, marker_color="#2dd4bf"))
                fig_pred.update_layout(title="Predicciones por columna", yaxis_title="Valor normalizado", paper_bgcolor="#0d0f14", plot_bgcolor="#141720", font={"color": "#e2e8f0"}, height=400)
                st.plotly_chart(fig_pred, use_container_width=True)

# ── Step 6: Test Model ────────────────────────────────────────────────────────
if st.session_state.get("dl_results"):
    last_result = st.session_state.dl_results[-1]
    if last_result["model_type"] == model_type:
        st.divider()
        st.markdown("### 🧪 Probar Modelo")
        st.caption("Selecciona una fila del dataset, edítala si lo deseas y prueba el modelo.")

        max_rows = min(1000, len(df))
        selected_idx = st.slider("Índice de fila", 0, max_rows - 1, 0, key="test_row_idx")
        
        relevant_cols = last_result.get("feature_cols", df.columns.tolist())
        test_row = df.iloc[selected_idx:selected_idx+1][relevant_cols].copy()
        
        st.markdown("#### Datos de entrada (editables)")
        edited_row = st.data_editor(test_row, num_rows="fixed", key="test_row_editor", use_container_width=True)
        
        if st.button("🚀 Probar Modelo", type="primary", use_container_width=True, key="test_model_btn"):
            try:
                model = last_result.get("model")
                scaler = last_result.get("scaler")
                
                # Prepare input
                row_vals = edited_row.values.astype(float)
                row_vals = np.nan_to_num(row_vals, nan=0.0)
                
                if scaler:
                    row_scaled = scaler.transform(row_vals)
                else:
                    row_scaled = row_vals

                if is_unsupervised:
                    x = torch.tensor(row_scaled, dtype=torch.float32)
                    with torch.no_grad():
                        recon = model(x).numpy()
                    
                    error = float(np.mean((row_scaled - recon) ** 2))
                    st.markdown("#### Resultado")
                    st.metric("Error de reconstrucción", f"{error:.6f}")
                    
                    # Anomaly check
                    train_errors = last_result.get("reconstruction_error")
                    if train_errors is not None:
                        p95 = float(np.percentile(train_errors, 95))
                        p99 = float(np.percentile(train_errors, 99))
                        if error > p99:
                            st.error(f"🔴 ANÓMALO (supera P99: {p99:.4f})")
                        elif error > p95:
                            st.warning(f"🟡 SOSPECHOSO (supera P95: {p95:.4f})")
                        else:
                            st.success("🟢 NORMAL")
                    
                    # Comparison table
                    comp_df = pd.DataFrame({
                        "Feature": relevant_cols,
                        "Original (scaled)": row_scaled.flatten(),
                        "Reconstruido (scaled)": recon.flatten(),
                        "Error": (row_scaled - recon).flatten() ** 2
                    })
                    st.dataframe(comp_df.round(4), use_container_width=True)
                    
                    # Plot
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Bar(x=comp_df["Feature"], y=comp_df["Original (scaled)"], name="Original", marker_color="#5b6af0"))
                    fig_comp.add_trace(go.Bar(x=comp_df["Feature"], y=comp_df["Reconstruido (scaled)"], name="Reconstruido", marker_color="#2dd4bf"))
                    fig_comp.update_layout(title="Original vs Reconstruido", barmode="group", paper_bgcolor="#0d0f14", plot_bgcolor="#141720", font={"color": "#e2e8f0"}, height=350)
                    st.plotly_chart(fig_comp, use_container_width=True)

                else:
                    # Sequence model
                    input_cols = last_result.get("input_cols", [])
                    predict_cols = last_result.get("predict_cols", [])
                    input_scaler = last_result.get("input_scaler")
                    
                    input_vals = edited_row[input_cols].values.astype(float).flatten()
                    input_vals = np.nan_to_num(input_vals, nan=0.0)
                    if input_scaler:
                        input_scaled = input_scaler.transform(input_vals.reshape(1, -1)).flatten()
                    else:
                        input_scaled = input_vals

                    x = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        preds = model(x, teacher_forcing_prob=0.0).numpy().flatten()
                    
                    st.markdown("#### Predicción Autoregresiva")
                    pred_df = pd.DataFrame({
                        "Columna a Predecir": predict_cols,
                        "Predicción (scaled)": preds.round(4)
                    })
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    
                    # Original values for comparison if available in dataset
                    if selected_idx + len(predict_cols) < len(df):
                        orig_vals = df.iloc[selected_idx + len(input_cols) : selected_idx + len(input_cols) + len(predict_cols)][predict_cols].values.flatten()
                        if scaler:
                            # Approximate inverse for display
                            orig_scaled = scaler.transform(df.iloc[selected_idx + len(input_cols) : selected_idx + len(input_cols) + len(predict_cols)][predict_cols].values.reshape(1, -1)).flatten()
                        else:
                            orig_scaled = orig_vals
                            
                        pred_df["Valor Real en Dataset (scaled)"] = orig_scaled.round(4)
                        pred_df["Diferencia"] = (pred_df["Predicción (scaled)"] - pred_df["Valor Real en Dataset (scaled)"]).round(4)
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No hay suficientes filas siguientes en el dataset para comparar con valores reales.")

            except Exception as e:
                st.error(f"Error al probar el modelo: {e}")
                import traceback
                st.code(traceback.format_exc())

        # ── Save Model Section ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### 💾 Guardar Modelo")
        st.caption("Guarda el modelo entrenado para usarlo en la sección **Test Models**.")

        if "saved_dl_models" not in st.session_state:
            st.session_state.saved_dl_models = {}

        model_name = st.text_input("Nombre del modelo", value=f"{model_type}_{len(st.session_state.saved_dl_models)+1}", key="save_model_name")
        
        col_save1, col_save2 = st.columns([1, 2])
        with col_save1:
            if st.button("💾 Guardar", type="primary", use_container_width=True, key="save_model_btn"):
                if model_name:
                    if model_name in st.session_state.saved_dl_models:
                        st.warning(f"Ya existe un modelo llamado '{model_name}'. Se sobrescribirá.")
                    
                    st.session_state.saved_dl_models[model_name] = {
                        "name": model_name,
                        "type": model_type,
                        "model": last_result.get("model"),
                        "scaler": last_result.get("scaler"),
                        "input_scaler": last_result.get("input_scaler"),
                        "feature_cols": last_result.get("feature_cols", []),
                        "config": last_result.get("config", {}),
                        "metadata": {
                            "train_rmse": last_result.get("train_rmse"),
                            "val_rmse": last_result.get("val_rmse"),
                            "reconstruction_error": last_result.get("reconstruction_error"),
                            "slots": last_result.get("slots"),
                            "predict_cols": last_result.get("predict_cols"),
                            "input_cols": last_result.get("input_cols"),
                            "history": last_result.get("history"),
                        },
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    st.success(f"✅ Modelo '{model_name}' guardado exitosamente.")
                    st.rerun()
        
        with col_save2:
            if st.session_state.saved_dl_models:
                st.markdown("**Modelos guardados:**")
                for name, info in st.session_state.saved_dl_models.items():
                    st.caption(f"• `{name}` ({info['type'].upper()}) - {info['timestamp']}")
