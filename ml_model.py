from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

LOGGER = logging.getLogger(__name__)


@dataclass
class ATRModelConfig:
    model_path: str = "artifacts/atr_model.pkl"
    random_state: int = 42


class ATRModel:
    def __init__(self, config: Optional[ATRModelConfig] = None):
        self.config = config or ATRModelConfig()
        self.model = RandomForestRegressor(n_estimators=200, random_state=self.config.random_state)
        self.feature_cols: List[str] = []

    def train(self, df: pd.DataFrame, feature_cols: List[str], target_col: str = "atr_target"):
        train_df = df.dropna(subset=feature_cols + [target_col]).copy()
        if train_df.empty:
            raise ValueError("Dados insuficientes para treinar o modelo de ATR.")

        self.feature_cols = feature_cols
        self.model.fit(train_df[feature_cols], train_df[target_col])
        LOGGER.info("Modelo de ATR treinado com %s amostras.", len(train_df))

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.feature_cols:
            raise ValueError("Modelo não treinado: feature_cols ausentes.")
        pred = self.model.predict(df[self.feature_cols].fillna(method="ffill").fillna(method="bfill"))
        return pd.Series(pred, index=df.index, name="atr_pred")

    def save(self):
        path = Path(self.config.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.model, "feature_cols": self.feature_cols}
        joblib.dump(payload, path)
        LOGGER.info("Modelo salvo em %s", path)

    def load(self):
        payload = joblib.load(self.config.model_path)
        self.model = payload["model"]
        self.feature_cols = payload["feature_cols"]
        LOGGER.info("Modelo carregado de %s", self.config.model_path)
