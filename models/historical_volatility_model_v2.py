import numpy as np
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class HistoricalVolatilityModelV2(BaseModel):
    """
    Наивный прогноз волатильности с адаптивным окном.
    Перебирает окна 10, 20, 60 дней и выбирает лучшее
    по минимальной ошибке на последних данных.
    """
    name = "HistoricalVolatilityModelV2"
    required_fields = ['returns']

    def __init__(self, seed=None):
        self.candidate_windows = [10, 20, 60]
        self.best_window = 20
        self.predicted_vol = 0.01

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean = returns[~np.isnan(returns)]

        if len(clean) < max(self.candidate_windows) + 10:
            if len(clean) > 0:
                self.predicted_vol = float(np.nanstd(clean))
            else:
                self.predicted_vol = 0.01
            return

        best_mse = float('inf')
        best_window = 20

        for window in self.candidate_windows:
            if len(clean) < window + 5:
                continue

            # Проверяем качество прогноза на последних 5 днях
            mse_sum = 0.0
            count = 0
            for i in range(len(clean) - 5, len(clean) - 1):
                if i >= window:
                    pred_vol = np.nanstd(clean[i - window:i])
                    actual_vol = abs(clean[i + 1]) if i + 1 < len(clean) else abs(clean[i])
                    mse_sum += (pred_vol - actual_vol) ** 2
                    count += 1

            if count > 0:
                mse = mse_sum / count
                if mse < best_mse:
                    best_mse = mse
                    best_window = window

        self.best_window = best_window

        if len(clean) >= best_window:
            self.predicted_vol = float(np.nanstd(clean[-best_window:]))
        else:
            self.predicted_vol = float(np.nanstd(clean))

        self.predicted_vol = max(self.predicted_vol, 1e-6)

    def predict(self, data: dict) -> dict:
        return {
            'predicted_volatility': float(self.predicted_vol),
            'annualized_volatility': float(self.predicted_vol * np.sqrt(252))
        }