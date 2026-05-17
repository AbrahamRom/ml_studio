"""Deep learning models: Autoencoder, VAE, LSTM, GRU for tabular data."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dims: list[int], dropout: float = 0.0):
        super().__init__()
        dims = [input_dim] + encoding_dims
        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            encoder_layers.append(nn.BatchNorm1d(dims[i + 1]))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        dec_dims = encoding_dims + [input_dim]
        for i in range(len(dec_dims) - 1):
            decoder_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                decoder_layers.append(nn.BatchNorm1d(dec_dims[i + 1]))
                decoder_layers.append(nn.ReLU())
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class VAE(nn.Module):
    def __init__(self, input_dim: int, encoding_dims: list[int], dropout: float = 0.0):
        super().__init__()
        dims = [input_dim] + encoding_dims
        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            encoder_layers.append(nn.BatchNorm1d(dims[i + 1]))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
        self.encoder_body = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoding_dims[-1], encoding_dims[-1])
        self.fc_logvar = nn.Linear(encoding_dims[-1], encoding_dims[-1])

        decoder_layers = []
        dec_dims = encoding_dims + [input_dim]
        for i in range(len(dec_dims) - 1):
            decoder_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                decoder_layers.append(nn.BatchNorm1d(dec_dims[i + 1]))
                decoder_layers.append(nn.ReLU())
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def encode(self, x):
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        return mu


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        task: str = "regression",
    ):
        super().__init__()
        self.task = task
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.fc(last_step)


class GRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        task: str = "regression",
    ):
        super().__init__()
        self.task = task
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_step = gru_out[:, -1, :]
        return self.fc(last_step)


# ──────────────────────────────────────────────────────────────────────────────
# Training loops
# ──────────────────────────────────────────────────────────────────────────────

def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict,
    progress_callback=None,
) -> dict:
    set_seed(config.get("random_state", 42))
    input_dim = X_train.shape[1]
    encoding_dims = config.get("encoding_dims", [64, 32, 16])
    lr = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 100)
    dropout = config.get("dropout", 0.0)
    patience = config.get("patience", 10)

    model = Autoencoder(input_dim, encoding_dims, dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=batch_size, shuffle=True)
    val_tensor = torch.FloatTensor(X_val)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_out = model(val_tensor.to(DEVICE))
            val_loss = criterion(val_out, val_tensor.to(DEVICE)).item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if progress_callback:
            progress_callback(epoch + 1, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    with torch.no_grad():
        train_encoded = model.encode(torch.FloatTensor(X_train)).cpu().numpy()
        val_encoded = model.encode(val_tensor).cpu().numpy()
        train_reconstructed = model(torch.FloatTensor(X_train)).cpu().numpy()
        val_reconstructed = model(val_tensor).cpu().numpy()

    train_rmse = float(np.sqrt(np.mean((X_train - train_reconstructed) ** 2)))
    val_rmse = float(np.sqrt(np.mean((X_val - val_reconstructed) ** 2)))

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_encoded": train_encoded,
        "val_encoded": val_encoded,
        "train_reconstructed": train_reconstructed,
        "val_reconstructed": val_reconstructed,
        "encoding_dims": encoding_dims,
    }


def train_vae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict,
    progress_callback=None,
) -> dict:
    set_seed(config.get("random_state", 42))
    input_dim = X_train.shape[1]
    encoding_dims = config.get("encoding_dims", [64, 32, 16])
    lr = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 100)
    dropout = config.get("dropout", 0.0)
    patience = config.get("patience", 10)
    kl_weight = config.get("kl_weight", 1.0)

    model = VAE(input_dim, encoding_dims, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=batch_size, shuffle=True)
    val_tensor = torch.FloatTensor(X_val)

    def vae_loss(recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kl_weight * kld) / x.size(0)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kld_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_recon, val_mu, val_logvar = model(val_tensor.to(DEVICE))
            val_loss = vae_loss(val_recon, val_tensor.to(DEVICE), val_mu, val_logvar).item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if progress_callback:
            progress_callback(epoch + 1, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    with torch.no_grad():
        train_encoded = model.encode(torch.FloatTensor(X_train)).cpu().numpy()
        val_encoded = model.encode(val_tensor).cpu().numpy()
        train_recon, _, _ = model(torch.FloatTensor(X_train))
        train_recon = train_recon.cpu().numpy()
        val_recon, _, _ = model(val_tensor)
        val_recon = val_recon.cpu().numpy()

    train_rmse = float(np.sqrt(np.mean((X_train - train_recon) ** 2)))
    val_rmse = float(np.sqrt(np.mean((X_val - val_recon) ** 2)))

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_encoded": train_encoded,
        "val_encoded": val_encoded,
        "train_reconstructed": train_recon,
        "val_reconstructed": val_recon,
        "encoding_dims": encoding_dims,
        "kl_weight": kl_weight,
    }


def _prepare_sequences(X: np.ndarray, y: np.ndarray | None, seq_length: int):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        if y is not None:
            y_seq.append(y[i + seq_length - 1])
    X_seq = np.array(X_seq)
    if y is not None:
        y_seq = np.array(y_seq)
        return X_seq, y_seq
    return X_seq, None


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    progress_callback=None,
) -> dict:
    set_seed(config.get("random_state", 42))
    task = config.get("task", "regression")
    input_dim = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]
    seq_length = config.get("seq_length", 5)
    hidden_dim = config.get("hidden_dim", 64)
    num_layers = config.get("num_layers", 2)
    lr = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 100)
    dropout = config.get("dropout", 0.1)
    patience = config.get("patience", 10)
    bidirectional = config.get("bidirectional", False)

    if X_train.ndim == 2:
        X_train, y_train = _prepare_sequences(X_train, y_train, seq_length)
        X_val, y_val = _prepare_sequences(X_val, y_val, seq_length)

    output_dim = 1 if task == "regression" else int(np.max(y_train)) + 1

    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout, bidirectional, task).to(DEVICE)

    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train) if task == "regression" else torch.LongTensor(y_train)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val) if task == "regression" else torch.LongTensor(y_val)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            output = model(xb)
            if task == "regression":
                loss = criterion(output.squeeze(), yb)
            else:
                loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t.to(DEVICE))
            if task == "regression":
                val_loss = criterion(val_out.squeeze(), y_val_t.to(DEVICE)).item()
            else:
                val_loss = criterion(val_out, y_val_t.to(DEVICE)).item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if progress_callback:
            progress_callback(epoch + 1, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t.to(DEVICE)).cpu().numpy()
        val_pred = model(X_val_t.to(DEVICE)).cpu().numpy()

    if task == "regression":
        train_rmse = float(np.sqrt(np.mean((y_train - train_pred.flatten()) ** 2)))
        val_rmse = float(np.sqrt(np.mean((y_val - val_pred.flatten()) ** 2)))
    else:
        train_pred_cls = np.argmax(train_pred, axis=1)
        val_pred_cls = np.argmax(val_pred, axis=1)
        train_rmse = float(1 - np.mean(y_train == train_pred_cls))
        val_rmse = float(1 - np.mean(y_val == val_pred_cls))

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_predictions": train_pred.flatten() if task == "regression" else np.argmax(train_pred, axis=1),
        "val_predictions": val_pred.flatten() if task == "regression" else np.argmax(val_pred, axis=1),
        "seq_length": seq_length,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }


def train_gru(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    progress_callback=None,
) -> dict:
    set_seed(config.get("random_state", 42))
    task = config.get("task", "regression")
    input_dim = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]
    seq_length = config.get("seq_length", 5)
    hidden_dim = config.get("hidden_dim", 64)
    num_layers = config.get("num_layers", 2)
    lr = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 100)
    dropout = config.get("dropout", 0.1)
    patience = config.get("patience", 10)
    bidirectional = config.get("bidirectional", False)

    if X_train.ndim == 2:
        X_train, y_train = _prepare_sequences(X_train, y_train, seq_length)
        X_val, y_val = _prepare_sequences(X_val, y_val, seq_length)

    output_dim = 1 if task == "regression" else int(np.max(y_train)) + 1

    model = GRUModel(input_dim, hidden_dim, num_layers, output_dim, dropout, bidirectional, task).to(DEVICE)

    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train) if task == "regression" else torch.LongTensor(y_train)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val) if task == "regression" else torch.LongTensor(y_val)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            output = model(xb)
            if task == "regression":
                loss = criterion(output.squeeze(), yb)
            else:
                loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t.to(DEVICE))
            if task == "regression":
                val_loss = criterion(val_out.squeeze(), y_val_t.to(DEVICE)).item()
            else:
                val_loss = criterion(val_out, y_val_t.to(DEVICE)).item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if progress_callback:
            progress_callback(epoch + 1, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t.to(DEVICE)).cpu().numpy()
        val_pred = model(X_val_t.to(DEVICE)).cpu().numpy()

    if task == "regression":
        train_rmse = float(np.sqrt(np.mean((y_train - train_pred.flatten()) ** 2)))
        val_rmse = float(np.sqrt(np.mean((y_val - val_pred.flatten()) ** 2)))
    else:
        train_pred_cls = np.argmax(train_pred, axis=1)
        val_pred_cls = np.argmax(val_pred, axis=1)
        train_rmse = float(1 - np.mean(y_train == train_pred_cls))
        val_rmse = float(1 - np.mean(y_val == val_pred_cls))

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_predictions": train_pred.flatten() if task == "regression" else np.argmax(train_pred, axis=1),
        "val_predictions": val_pred.flatten() if task == "regression" else np.argmax(val_pred, axis=1),
        "seq_length": seq_length,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Unified interface
# ──────────────────────────────────────────────────────────────────────────────

DL_MODEL_TYPES = ["autoencoder", "vae", "lstm", "gru"]

DL_DEFAULT_CONFIGS = {
    "autoencoder": {
        "encoding_dims": [64, 32, 16],
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "dropout": 0.0,
        "patience": 10,
    },
    "vae": {
        "encoding_dims": [64, 32, 16],
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "dropout": 0.0,
        "patience": 10,
        "kl_weight": 1.0,
    },
    "lstm": {
        "hidden_dim": 64,
        "num_layers": 2,
        "seq_length": 5,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "dropout": 0.1,
        "patience": 10,
        "bidirectional": False,
    },
    "gru": {
        "hidden_dim": 64,
        "num_layers": 2,
        "seq_length": 5,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "dropout": 0.1,
        "patience": 10,
        "bidirectional": False,
    },
}


def run_dl_model(
    df: pd.DataFrame,
    model_type: str,
    target: str | None = None,
    config: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    progress_callback=None,
) -> dict:
    """Train a deep learning model and return results."""

    if model_type not in DL_MODEL_TYPES:
        raise ValueError(f"Model type '{model_type}' not supported. Choose from {DL_MODEL_TYPES}")

    set_seed(random_state)
    cfg = {**DL_DEFAULT_CONFIGS[model_type], **(config or {})}
    cfg["random_state"] = random_state

    feature_cols = [c for c in df.columns if c != target] if target else df.columns.tolist()
    if not feature_cols:
        raise ValueError("No feature columns available.")

    data = df.loc[df[target].notna(), feature_cols + ([target] if target else [])].copy() if target else df[feature_cols].copy()

    if target:
        if data[target].nunique(dropna=True) < 2:
            raise ValueError(f"Target '{target}' has fewer than 2 distinct non-null values.")

    X = data[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if target:
        y = data[target].values
        if pd.api.types.is_numeric_dtype(data[target]):
            y = y.astype(np.float32)
            task = "regression"
        else:
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            task = "classification"
            cfg["task"] = "classification"

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
    else:
        task = "unsupervised"
        X_train, X_val = train_test_split(X_scaled, test_size=test_size, random_state=random_state)
        y_train = y_val = None

    if model_type == "autoencoder":
        result = train_autoencoder(X_train, X_val, cfg, progress_callback)
    elif model_type == "vae":
        result = train_vae(X_train, X_val, cfg, progress_callback)
    elif model_type == "lstm":
        result = train_lstm(X_train, y_train, X_val, y_val, cfg, progress_callback)
    elif model_type == "gru":
        result = train_gru(X_train, y_train, X_val, y_val, cfg, progress_callback)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    result["model_type"] = model_type
    result["target"] = target
    result["feature_cols"] = feature_cols
    result["scaler"] = scaler
    result["task"] = task
    result["train_rows"] = len(X_train)
    result["val_rows"] = len(X_val)
    result["config"] = cfg

    if target and not pd.api.types.is_numeric_dtype(data[target]):
        result["label_encoder"] = le

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter optimization (random search)
# ──────────────────────────────────────────────────────────────────────────────

HPO_SEARCH_SPACES = {
    "autoencoder": {
        "encoding_dims": [
            [128, 64, 32],
            [64, 32, 16],
            [64, 32],
            [128, 64, 32, 16],
            [256, 128, 64],
            [32, 16],
        ],
        "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [16, 32, 64, 128],
        "dropout": [0.0, 0.1, 0.2, 0.3],
    },
    "vae": {
        "encoding_dims": [
            [128, 64, 32],
            [64, 32, 16],
            [64, 32],
            [128, 64, 32, 16],
            [256, 128, 64],
        ],
        "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [16, 32, 64, 128],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "kl_weight": [0.1, 0.5, 1.0, 2.0, 5.0],
    },
    "lstm": {
        "hidden_dim": [32, 64, 128, 256],
        "num_layers": [1, 2, 3, 4],
        "seq_length": [3, 5, 7, 10, 15],
        "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [16, 32, 64],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
        "bidirectional": [True, False],
    },
    "gru": {
        "hidden_dim": [32, 64, 128, 256],
        "num_layers": [1, 2, 3, 4],
        "seq_length": [3, 5, 7, 10, 15],
        "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [16, 32, 64],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
        "bidirectional": [True, False],
    },
}


def _build_grid(model_type: str, base_config: dict, input_dim: int, n_samples: int) -> list[dict]:
    """Build a grid of hyperparameter combinations for grid search."""
    space = HPO_SEARCH_SPACES[model_type]

    if model_type in ("autoencoder", "vae"):
        valid_enc_dims = []
        for dims in space["encoding_dims"]:
            if all(d < input_dim for d in dims) and len(dims) >= 1:
                valid_enc_dims.append(dims)
        if not valid_enc_dims:
            valid_enc_dims = [[max(2, input_dim // 4)]]

        grid_items = {
            "encoding_dims": valid_enc_dims,
            "learning_rate": space["learning_rate"],
            "batch_size": [bs for bs in space["batch_size"] if bs <= n_samples],
            "dropout": space["dropout"],
        }
        if model_type == "vae":
            grid_items["kl_weight"] = space["kl_weight"]

    else:
        max_seq = min(max(space["seq_length"]), max(2, n_samples // 2))
        valid_seq = [s for s in space["seq_length"] if s <= max_seq]
        if not valid_seq:
            valid_seq = [2]

        grid_items = {
            "hidden_dim": space["hidden_dim"],
            "num_layers": space["num_layers"],
            "seq_length": valid_seq,
            "learning_rate": space["learning_rate"],
            "batch_size": [bs for bs in space["batch_size"] if bs <= n_samples],
            "dropout": space["dropout"],
            "bidirectional": space["bidirectional"],
        }

    keys = list(grid_items.keys())
    values = list(grid_items.values())
    import itertools
    combinations = list(itertools.product(*values))

    result = []
    for combo in combinations:
        config = dict(zip(keys, combo))
        for key, val in base_config.items():
            if key not in space and key not in ("random_state", "epochs", "patience"):
                config[key] = val
        result.append(config)

    return result


def optimize_dl_hyperparameters(
    df: pd.DataFrame,
    model_type: str,
    target: str | None = None,
    base_config: dict | None = None,
    n_trials: int = 20,
    epochs_per_trial: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    progress_callback=None,
) -> dict:
    """Grid search over hyperparameter space for deep learning models."""

    if model_type not in DL_MODEL_TYPES:
        raise ValueError(f"Model type '{model_type}' not supported.")

    cfg = {**DL_DEFAULT_CONFIGS[model_type], **(base_config or {})}
    cfg["epochs"] = epochs_per_trial
    cfg["random_state"] = random_state
    cfg["patience"] = max(5, epochs_per_trial // 5)

    feature_cols = [c for c in df.columns if c != target] if target else df.columns.tolist()
    data = df.loc[df[target].notna(), feature_cols + ([target] if target else [])].copy() if target else df[feature_cols].copy()

    X = data[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if target:
        y = data[target].values
        if pd.api.types.is_numeric_dtype(data[target]):
            y = y.astype(np.float32)
            task = "regression"
        else:
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            task = "classification"
            cfg["task"] = "classification"
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    else:
        task = "unsupervised"
        X_train, X_val = train_test_split(X_scaled, test_size=test_size, random_state=random_state)
        y_train = y_val = None

    input_dim = X_train.shape[-1]
    n_train = len(X_train)

    grid = _build_grid(model_type, cfg, input_dim, n_train)
    grid_configs = []
    for params in grid:
        full_cfg = {**cfg, **params}
        full_cfg["epochs"] = epochs_per_trial
        full_cfg["patience"] = max(5, epochs_per_trial // 5)
        grid_configs.append(full_cfg)

    if len(grid_configs) > n_trials:
        import itertools
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(grid_configs), size=n_trials, replace=False)
        grid_configs = [grid_configs[i] for i in sorted(indices)]

    trials = []
    best_val_loss = float("inf")
    best_trial_idx = -1
    total_trials = len(grid_configs)

    for trial, trial_cfg in enumerate(grid_configs):
        trial_cfg["random_state"] = random_state + trial
        trial_num = trial

        def trial_progress(epoch, total, train_loss, val_loss, _tn=trial_num):
            if progress_callback:
                progress_callback(_tn, total_trials, epoch, total, train_loss, val_loss)

        try:
            if model_type == "autoencoder":
                result = train_autoencoder(X_train, X_val, trial_cfg, trial_progress)
            elif model_type == "vae":
                result = train_vae(X_train, X_val, trial_cfg, trial_progress)
            elif model_type == "lstm":
                result = train_lstm(X_train, y_train, X_val, y_val, trial_cfg, trial_progress)
            elif model_type == "gru":
                result = train_gru(X_train, y_train, X_val, y_val, trial_cfg, trial_progress)
            else:
                continue

            val_loss = result["best_val_loss"]
            if not np.isfinite(val_loss):
                raise ValueError(f"Non-finite val_loss: {val_loss}")

            trial_info = {
                "trial": trial + 1,
                "val_loss": val_loss,
                "train_rmse": result["train_rmse"],
                "val_rmse": result["val_rmse"],
                "config": {k: v for k, v in trial_cfg.items() if k not in ("random_state", "epochs", "patience")},
                "history": result["history"],
            }
            trials.append(trial_info)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_trial_idx = trial

        except Exception as e:
            trials.append({
                "trial": trial + 1,
                "val_loss": float("inf"),
                "train_rmse": float("inf"),
                "val_rmse": float("inf"),
                "config": {k: v for k, v in trial_cfg.items() if k not in ("random_state", "epochs", "patience")},
                "error": True,
                "error_msg": str(e),
            })

    best_trial = trials[best_trial_idx] if best_trial_idx >= 0 else trials[0]
    best_config = best_trial["config"]
    best_config["epochs"] = cfg.get("epochs", 100)
    best_config["patience"] = cfg.get("patience", 10)
    best_config["random_state"] = random_state

    return {
        "best_config": best_config,
        "best_val_loss": best_val_loss,
        "trials": trials,
        "n_trials": total_trials,
        "model_type": model_type,
        "target": target,
        "task": task,
        "feature_cols": feature_cols,
    }
