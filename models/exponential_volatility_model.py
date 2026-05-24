import numpy as np
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class ExponentialVolatilityModel(BaseModel):
    """
    Экспоненциально взвешенная волатильность (EWMA).
    Стандарт RiskMetrics: alpha = 0.94, окно ~ 20 дней эффективной памяти.
    Быстро адаптируется к всплескам волатильности.
    """
    name = "ExponentialVolatilityModel"
    required_fields = ['returns']

    def __init__(self, seed=None, alpha=0.94, window=60):
        """
        alpha: коэффициент забывания (0..1). 0.94 = RiskMetrics standard.
        window: сколько последних дней использовать для расчёта.
        """
        self.alpha = alpha
        self.window = window
        self.predicted_vol = 0.01

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean = returns[~np.isnan(returns)]

        if len(clean) < 10:
            self.predicted_vol = float(np.nanstd(clean)) if len(clean) > 0 else 0.01
            return

        # Берём последние window дней (или меньше)
        n = min(len(clean), self.window)
        recent = clean[-n:]

        # Экспоненциальные веса
        weights = np.power(self.alpha, np.arange(n)[::-1])
        weights /= weights.sum()

        # Средневзвешенное квадратов доходностей (центрированное)
        weighted_mean = np.average(recent, weights=weights)
        weighted_var = np.average((recent - weighted_mean) ** 2, weights=weights)

        self.predicted_vol = float(np.sqrt(weighted_var))
        self.predicted_vol = max(self.predicted_vol, 1e-6)

    def predict(self, data: dict) -> dict:
        return {
            'predicted_volatility': float(self.predicted_vol),
            'annualized_volatility': float(self.predicted_vol * np.sqrt(252))
        }