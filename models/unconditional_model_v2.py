import numpy as np
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class UnconditionalModelV2(BaseModel):
    """
    Безусловная модель с экспоненциальным взвешиванием.
    Недавние дни имеют больший вес при подсчёте вероятности роста.
    """
    name = "UnconditionalModelV2"
    required_fields = ['returns']

    def __init__(self, seed=None, alpha=0.94):
        """
        alpha: коэффициент забывания (0..1).
        0.94 — стандарт RiskMetrics.
        Чем меньше alpha, тем быстрее забываем старые данные.
        """
        self.alpha = alpha
        self.prob_up = 0.5

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean = returns[~np.isnan(returns)]

        if len(clean) == 0:
            self.prob_up = 0.5
            return

        # Экспоненциальные веса: w[i] = alpha^(n-1-i) для i-го элемента с конца
        n = len(clean)
        weights = np.power(self.alpha, np.arange(n)[::-1])
        weights /= weights.sum()

        # Взвешенная доля положительных доходностей
        positive = (clean > 0).astype(float)
        self.prob_up = float(np.average(positive, weights=weights))
        self.prob_up = np.clip(self.prob_up, 0.001, 0.999)

    def predict(self, data: dict) -> dict:
        direction = 1 if self.prob_up >= 0.5 else 0

        return {
            'direction': int(direction),
            'prob_up': float(self.prob_up)
        }