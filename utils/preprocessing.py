import pandas as pd
import numpy as np
from typing import Tuple


class RobustScaler:
    """Estandarización robusta usando mediana e IQR.
    
    Fórmula: (x - mediana) / IQR
    
    Resistente a outliers ya que no usa media ni desviación estándar.
    Compatible con la interfaz de sklearn (fit, transform, fit_transform).
    """

    def __init__(self):
        self.median_ = None
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None

    def fit(self, X: pd.DataFrame) -> "RobustScaler":
        self.median_ = X.median()
        self.q1_ = X.quantile(0.25)
        self.q3_ = X.quantile(0.75)
        self.iqr_ = self.q3_ - self.q1_
        self.iqr_ = self.iqr_.replace(0, 1)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.median_ is None:
            raise ValueError("Este RobustScaler no ha sido ajustado. Llama a fit() primero.")
        return (X - self.median_) / self.iqr_

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        if self.median_ is None:
            raise ValueError("Este RobustScaler no ha sido ajustado. Llama a fit() primero.")
        return X_scaled * self.iqr_ + self.median_

    def get_params(self) -> dict:
        return {
            "median": self.median_.to_dict() if self.median_ is not None else None,
            "q1": self.q1_.to_dict() if self.q1_ is not None else None,
            "q3": self.q3_.to_dict() if self.q3_ is not None else None,
            "iqr": self.iqr_.to_dict() if self.iqr_ is not None else None,
        }


def get_scaler(scaler_type: str = "robust"):
    """Factory para obtener escaladores compatibles con pandas DataFrames.
    
    Args:
        scaler_type: "robust", "standard", o "minmax"
    
    Returns:
        Instancia del escalador seleccionado.
    """
    if scaler_type == "robust":
        return RobustScaler()
    elif scaler_type == "standard":
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    elif scaler_type == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler()
    else:
        raise ValueError(f"Tipo de escalador desconocido: {scaler_type}. Usa 'robust', 'standard' o 'minmax'.")
