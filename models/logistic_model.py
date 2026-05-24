import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class LogisticModel(BaseModel):
    name = "LogisticModel"
    required_fields = ['returns', 'close']

    def __init__(self, seed=None):
        self.model = LogisticRegression(
            penalty=None,
            solver='lbfgs',
            max_iter=500,
            random_state=seed
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            'return_t',
            'return_lag1',
            'volatility_5',
            'volatility_20',
            'rsi_14',
            'momentum_5',
        ]
        self.fitted = False

    def _compute_rsi(self, close, window=14):
        if len(close) < window + 1:
            return np.full(len(close), np.nan)

        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.full(len(close), np.nan)
        avg_loss = np.full(len(close), np.nan)

        avg_gain[window] = gain[1:window+1].mean()
        avg_loss[window] = loss[1:window+1].mean()

        for i in range(window + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (window - 1) + gain[i]) / window
            avg_loss[i] = (avg_loss[i-1] * (window - 1) + loss[i]) / window

        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _build_features(self, data: dict):
        returns = np.asarray(data['returns'], dtype=float)
        close = np.asarray(data['close'], dtype=float)

        n = len(returns)
        if n < 30:
            return None, None

        features = {}
        features['return_t'] = returns
        features['return_lag1'] = np.roll(returns, 1)
        features['return_lag1'][0] = np.nan

        features['volatility_5'] = np.array([
            np.nanstd(returns[max(0, i-4):i+1]) if i >= 4 else np.nan
            for i in range(n)
        ])
        features['volatility_20'] = np.array([
            np.nanstd(returns[max(0, i-19):i+1]) if i >= 19 else np.nan
            for i in range(n)
        ])

        features['rsi_14'] = self._compute_rsi(close, window=14)

        momentum = np.full(n, np.nan)
        for i in range(5, n):
            momentum[i] = close[i] / close[i-5] - 1
        features['momentum_5'] = momentum

        X = np.column_stack([features[name] for name in self.feature_names])
        Y = np.where(np.roll(returns, -1) > 0, 1, 0)

        valid_mask = np.arange(n) >= 25
        valid_mask &= ~np.isnan(X).any(axis=1)
        valid_mask &= ~np.isnan(Y)

        X = X[valid_mask]
        Y = Y[valid_mask]

        if len(X) < 10:
            return None, None

        return X, Y

    def fit(self, data: dict) -> None:
        X, Y = self._build_features(data)

        if X is None or len(X) < 10:
            self.fitted = False
            return

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, Y)
            self.fitted = True
        except Exception:
            self.fitted = False

    def predict(self, data: dict) -> dict:
        if not self.fitted:
            return {'direction': 1, 'prob_up': 0.5}

        X, _ = self._build_features(data)
        if X is None or len(X) == 0:
            return {'direction': 1, 'prob_up': 0.5}

        last_X = X[-1:]

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                last_X_scaled = self.scaler.transform(last_X)
                prob_up = self.model.predict_proba(last_X_scaled)[0, 1]
            prob_up = np.clip(prob_up, 0.001, 0.999)
            direction = 1 if prob_up >= 0.5 else 0

            return {
                'direction': int(direction),
                'prob_up': float(prob_up)
            }
        except Exception:
            return {'direction': 1, 'prob_up': 0.5}