import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class LogisticModelV2(BaseModel):
    name = "LogisticModelV2"
    required_fields = ['returns', 'close', 'dates']

    def __init__(self, seed=None):
        self.model = LogisticRegression(
            penalty='l2',
            C=10.0,
            solver='lbfgs',
            max_iter=500,
            random_state=seed
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            'return_t', 'return_lag1',
            'volatility_5', 'volatility_20',
            'rsi_14', 'momentum_5',
            'baseline_prob', 'day_of_week',
            'spread_ratio', 'volume_ratio',
            'pct_from_high_20', 'rsi_vol_interaction',
        ]
        self.fitted = False

    def _compute_rsi(self, close, window=14):
        n = len(close)
        if n < window + 1:
            return np.full(n, np.nan)
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        avg_gain[window] = gain[1:window+1].mean()
        avg_loss[window] = loss[1:window+1].mean()
        for i in range(window + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (window - 1) + gain[i]) / window
            avg_loss[i] = (avg_loss[i-1] * (window - 1) + loss[i]) / window
        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        return 100 - (100 / (1 + rs))

    def _build_features(self, data: dict):
        returns = np.asarray(data['returns'], dtype=float)
        close = np.asarray(data['close'], dtype=float)
        dates = data['dates']

        n = len(returns)
        if n < 30:
            return None, None

        return_t = returns.copy()
        return_lag1 = np.roll(returns, 1); return_lag1[0] = np.nan

        vol5 = np.full(n, np.nan)
        for i in range(n):
            if i >= 4: vol5[i] = np.nanstd(returns[i-4:i+1])

        vol20 = np.full(n, np.nan)
        for i in range(n):
            if i >= 19: vol20[i] = np.nanstd(returns[i-19:i+1])

        rsi = self._compute_rsi(close, window=14)

        momentum = np.full(n, np.nan)
        for i in range(5, n): momentum[i] = close[i]/close[i-5] - 1

        clean_ret = returns[~np.isnan(returns)]
        baseline = (clean_ret > 0).mean() if len(clean_ret) > 0 else 0.5
        baseline_prob = np.full(n, baseline)

        dow_arr = np.zeros(n)
        if hasattr(dates, 'dayofweek'):
            dvals = dates.dayofweek.values if hasattr(dates.dayofweek, 'values') else np.array(dates.dayofweek)
            dvals = dvals[:n]
            dvals = np.where(dvals < 5, dvals, 4)
            dow_arr[:len(dvals)] = dvals

        spread = np.full(n, np.nan)
        for i in range(1, n): spread[i] = abs(close[i]-close[i-1])/close[i]

        volume_ratio = np.ones(n)

        pct_high = np.full(n, np.nan)
        for i in range(n):
            lb = min(i+1, 20)
            hi = np.max(close[i-lb+1:i+1])
            pct_high[i] = (close[i]-hi)/hi if hi != 0 else 0.0

        rsi_vol = np.full(n, np.nan)
        for i in range(n):
            v, r = vol20[i], rsi[i]
            if not np.isnan(r) and not np.isnan(v) and v > 1e-10:
                rsi_vol[i] = r/v

        feature_arrays = [
            return_t, return_lag1, vol5, vol20, rsi, momentum,
            baseline_prob, dow_arr, spread, volume_ratio, pct_high, rsi_vol
        ]

        X = np.column_stack(feature_arrays)
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
            return {'direction': int(direction), 'prob_up': float(prob_up)}
        except Exception:
            return {'direction': 1, 'prob_up': 0.5}