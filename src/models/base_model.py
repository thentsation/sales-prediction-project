from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ModelInterface(ABC):
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> tuple[float, float]: ...
