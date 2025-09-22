from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge


class RidgePerChannel:
    """
    Fits independent ridge regressors for each output channel.
    - X: (N, D)
    - Y: (N, C) (C = bands*4 in packed space)
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.models = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        C = Y.shape[1]
        self.models = [Ridge(alpha=self.alpha, fit_intercept=True) for _ in range(C)]
        for c in range(C):
            self.models[c].fit(X, Y[:, c])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.models is not None
        preds = [m.predict(X) for m in self.models]
        return np.stack(preds, axis=1)

