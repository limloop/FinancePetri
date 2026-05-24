import numpy as np
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class HistoricalVolatilityModel(BaseModel):
    """
    Наивный прогноз волатильности:
    предсказывает, что завтрашняя волатильность = средняя за последние window дней.
    """
    name = "HistoricalVolatilityModel"
    required_fields = ['returns']

    def __init__(self, seed=None, window=20):
        self.window = window
        self.predicted_vol = 0.01  # дефолт 1%

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean = returns[~np.isnan(returns)]

        if len(clean) >= self.window:
            self.predicted_vol = float(np.nanstd(clean[-self.window:]))
        elif len(clean) > 0:
            self.predicted_vol = float(np.nanstd(clean))
        else:
            self.predicted_vol = 0.01

        # Защита от нуля
        self.predicted_vol = max(self.predicted_vol, 1e-6)

    def predict(self, data: dict) -> dict:
        return {
            'predicted_volatility': float(self.predicted_vol),
            'annualized_volatility': float(self.predicted_vol * np.sqrt(252))
        }