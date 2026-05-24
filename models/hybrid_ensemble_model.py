import numpy as np
from models.base_model import BaseModel
from core.registry import registry
from core.market_regime import MarketRegimeDetector


@registry.register_model
class HybridEnsembleModel(BaseModel):
    """
    Гибридный ансамбль без ML.
    Комбинирует 3 модели направления и 2 модели волатильности
    с весами, зависящими от режима рынка.
    """
    name = "HybridEnsembleModel"
    required_fields = ['returns', 'close']

    def __init__(self, seed=None):
        self.detector = MarketRegimeDetector()

        # Веса для моделей направления в разных режимах
        self.direction_weights = {
            'trend_up':    [0.5, 0.2, 0.3],  # доверяем среднему
            'trend_down':  [0.5, 0.2, 0.3],  # доверяем среднему
            'volatile':    [0.2, 0.5, 0.3],  # доверяем V3
            'sideways':    [0.2, 0.4, 0.4],  # V3 + ансамбль
            'unknown':     [0.4, 0.3, 0.3],  # равномерно с креном к среднему
        }

        # Веса для моделей волатильности (фиксированные)
        self.volatility_weights = [0.5, 0.5]  # HistoricalV2, Exponential

        # Прогнозы
        self.prob_up = 0.5
        self.predicted_vol = 0.01
        self.regime = 'unknown'

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean = returns[~np.isnan(returns)]

        if len(clean) < 20:
            self.prob_up = 0.5
            self.predicted_vol = 0.01
            return

        # Определяем режим рынка
        regime_info = self.detector.detect(returns)
        self.regime = regime_info['regime']

        # === Модели направления ===
        n = len(clean)

        # 1. UnconditionalModel: простая доля ростов
        prob_unconditional = float((clean > 0).mean())

        # 2. UnconditionalModelV3: режимное снижение уверенности
        prob_raw = prob_unconditional
        trend_strength = abs(prob_raw - 0.5) * 2
        vol_ratio = regime_info['metrics'].get('vol_ratio', 1.0)

        confidence = 1.0
        if vol_ratio > 1.5:
            confidence *= 0.5
        if trend_strength < 0.2:
            confidence *= 0.5
        prob_v3 = 0.5 + (prob_raw - 0.5) * confidence

        # 3. UnconditionalEnsemble: ансамбль окон 20/60/100
        windows = [20, 60, 100]
        probs = []
        weights = []
        for w in windows:
            if n >= w:
                recent = clean[-w:]
                probs.append(float((recent > 0).mean()))
                weights.append(np.sqrt(w))
        if probs:
            prob_ensemble = float(np.average(probs, weights=weights))
        else:
            prob_ensemble = prob_unconditional

        # Взвешенное среднее
        dir_weights = self.direction_weights.get(self.regime, [0.4, 0.3, 0.3])
        self.prob_up = (
            dir_weights[0] * prob_unconditional +
            dir_weights[1] * prob_v3 +
            dir_weights[2] * prob_ensemble
        )
        self.prob_up = np.clip(self.prob_up, 0.001, 0.999)

        # === Модели волатильности ===

        # 1. HistoricalVolatilityModelV2: адаптивное окно
        candidate_windows = [10, 20, 60]
        best_window = 20
        best_mse = float('inf')

        for window in candidate_windows:
            if n < window + 5:
                continue
            mse_sum = 0.0
            count = 0
            for i in range(n - 5, n - 1):
                if i >= window:
                    pred_vol = np.nanstd(clean[i - window:i])
                    actual_vol = abs(clean[i + 1]) if i + 1 < n else abs(clean[i])
                    mse_sum += (pred_vol - actual_vol) ** 2
                    count += 1
            if count > 0:
                mse = mse_sum / count
                if mse < best_mse:
                    best_mse = mse
                    best_window = window

        if n >= best_window:
            vol_historical_v2 = float(np.nanstd(clean[-best_window:]))
        else:
            vol_historical_v2 = float(np.nanstd(clean))

        # 2. ExponentialVolatilityModel: EWMA
        ewma_n = min(n, 60)
        recent_ewma = clean[-ewma_n:]
        alpha = 0.94
        ewma_weights_arr = np.power(alpha, np.arange(ewma_n)[::-1])
        ewma_weights_arr /= ewma_weights_arr.sum()
        weighted_mean = np.average(recent_ewma, weights=ewma_weights_arr)
        weighted_var = np.average((recent_ewma - weighted_mean) ** 2, weights=ewma_weights_arr)
        vol_exponential = float(np.sqrt(weighted_var))

        # Взвешенное среднее
        self.predicted_vol = (
            self.volatility_weights[0] * vol_historical_v2 +
            self.volatility_weights[1] * vol_exponential
        )
        self.predicted_vol = max(self.predicted_vol, 1e-6)

    def predict(self, data: dict) -> dict:
        direction = 1 if self.prob_up >= 0.5 else 0

        return {
            'direction': int(direction),
            'prob_up': float(self.prob_up),
            'predicted_volatility': float(self.predicted_vol),
            'annualized_volatility': float(self.predicted_vol * np.sqrt(252)),
            'regime': self.regime
        }