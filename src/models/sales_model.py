import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .base_model import ModelInterface

DEFAULT_MODEL_PATH = "models/sales_predictor.pkl"


class SalesPredictor(ModelInterface):
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.logger = logger or logging.getLogger("SalesPrediction")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.logger.info("SalesPredictor model trained")

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float]:
        predictions = self.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        self.logger.info("SalesPredictor evaluated (MAE: %.4f, MSE: %.4f)", mae, mse)
        return mae, mse

    def save_model(self, path: str = DEFAULT_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Dumps the whole predictor (model + fitted scaler): loading only the
        # inner sklearn model at inference time would skip scaling and silently
        # return wrong predictions.
        joblib.dump(self, path)
        self.logger.info("SalesPredictor model saved to %s", path)

    @staticmethod
    def load(path: str = DEFAULT_MODEL_PATH) -> "SalesPredictor":
        predictor = joblib.load(path)
        if not isinstance(predictor, SalesPredictor):
            raise TypeError(f"{path} does not contain a SalesPredictor")
        return predictor
