import numpy as np
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class UnconditionalEnsemble(BaseModel):
    """
    Ансамбль безусловных моделей с разной глубиной памяти.
    Усредняет прогнозы короткой, средней и длинной памяти.
    """
    name = "UnconditionalEnsemble"
    required_fields = ['returns']

    def __init__(self, seed=None, windows=None):
        """
        windows: список размеров окон для ансамбля.
        По умолчанию: 20 (короткая), 60 (средняя), 100 (длинная) дней.
        """
        self.windows = windows or [20, 60, 100]
        self.prob_up = 0.5

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean = returns[~np.isnan(returns)]

        if len(clean) < min(self.windows):
            self.prob_up = 0.5
            return

        probs = []
        weights = []

        for window in self.windows:
            if len(clean) >= window:
                # Берём последние window дней
                recent = clean[-window:]
                prob = float((recent > 0).mean())
                # Вес обратно пропорционален стандартной ошибке: больше окно → точнее оценка
                weight = np.sqrt(window)
                probs.append(prob)
                weights.append(weight)

        if probs:
            self.prob_up = float(np.average(probs, weights=weights))
            self.prob_up = np.clip(self.prob_up, 0.001, 0.999)
        else:
            self.prob_up = 0.5

    def predict(self, data: dict) -> dict:
        direction = 1 if self.prob_up >= 0.5 else 0

        return {
            'direction': int(direction),
            'prob_up': float(self.prob_up)
        }