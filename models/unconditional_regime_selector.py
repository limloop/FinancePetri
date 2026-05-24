import numpy as np
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class UnconditionalRegimeSelector(BaseModel):
    """
    Выбирает стратегию в зависимости от режима рынка:
      - trend_up/trend_down → UnconditionalModel (простая доля ростов)
      - volatile → UnconditionalModelV3 (режимный детектор снижает уверенность)
      - sideways → UnconditionalEnsemble (ансамбль трёх окон)
      - unknown → UnconditionalModel (безопасный дефолт)
    """
    name = "UnconditionalRegimeSelector"
    required_fields = ['returns', 'regime']

    def __init__(self, seed=None):
        self.prob_up = 0.5

    def fit(self, data: dict) -> None:
        returns = data['returns']
        regime = data.get('regime', 'unknown')
        clean = returns[~np.isnan(returns)]

        if len(clean) < 20:
            self.prob_up = 0.5
            return

        if regime in ('trend_up', 'trend_down'):
            # Стратегия: простая доля ростов (как UnconditionalModel)
            self.prob_up = float((clean > 0).mean())

        elif regime == 'volatile':
            # Стратегия: режимный детектор (как V3)
            prob_up_raw = float((clean > 0).mean())
            trend_strength = abs(prob_up_raw - 0.5) * 2
            vol_ratio = data.get('regime_metrics', {}).get('vol_ratio', 1.0)

            confidence = 1.0
            if vol_ratio > 1.5:
                confidence *= 0.5
            if trend_strength < 0.2:
                confidence *= 0.5

            self.prob_up = 0.5 + (prob_up_raw - 0.5) * confidence

        elif regime == 'sideways':
            # Стратегия: ансамбль окон 20/60/100
            windows = [20, 60, 100]
            probs = []
            weights = []
            for w in windows:
                if len(clean) >= w:
                    recent = clean[-w:]
                    probs.append(float((recent > 0).mean()))
                    weights.append(np.sqrt(w))
            if probs:
                self.prob_up = float(np.average(probs, weights=weights))
            else:
                self.prob_up = float((clean > 0).mean())

        else:  # unknown
            self.prob_up = float((clean > 0).mean())

        self.prob_up = np.clip(self.prob_up, 0.001, 0.999)

    def predict(self, data: dict) -> dict:
        direction = 1 if self.prob_up >= 0.5 else 0
        return {
            'direction': int(direction),
            'prob_up': float(self.prob_up),
            'regime': data.get('regime', 'unknown')
        }