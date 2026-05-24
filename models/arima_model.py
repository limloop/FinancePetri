import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class ArimaModel(BaseModel):
    name = "ArimaModel"
    required_fields = ['returns']

    def __init__(self, order=(2, 0, 1), suppress_warnings=True):
        """
        order: (p, d, q) — параметры ARIMA.
        suppress_warnings: скрывать ли предупреждения statsmodels.
        """
        self.order = order
        self.suppress_warnings = suppress_warnings
        self.fitted_model = None

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean_returns = returns[~np.isnan(returns)]

        if len(clean_returns) < 20:
            self.fitted_model = None
            return

        # Пробуем основной порядок, при ошибке — упрощаем
        orders_to_try = [
            self.order,
            (1, 0, 0),   # AR(1) — самый простой и устойчивый
        ]

        for order in orders_to_try:
            try:
                with warnings.catch_warnings():
                    if self.suppress_warnings:
                        warnings.filterwarnings('ignore')

                    model = ARIMA(
                        clean_returns,
                        order=order,
                        enforce_stationarity=False,    # не требуем стационарность
                        enforce_invertibility=False    # не требуем обратимость
                    )
                    self.fitted_model = model.fit(method_kwargs={'maxiter': 100})
                return  # получилось — выходим
            except Exception:
                continue

        # Совсем ничего не вышло
        self.fitted_model = None

    def predict(self, data: dict) -> dict:
        if self.fitted_model is None:
            return {'direction': 1, 'prob_up': 0.5}

        try:
            forecast_result = self.fitted_model.get_forecast(steps=1)
            predicted_return = forecast_result.predicted_mean[0]
            se = forecast_result.se_mean[0]

            # Защита: если стандартная ошибка нулевая или отрицательная
            if se <= 0 or np.isnan(se):
                se = np.std(data['returns'][~np.isnan(data['returns'])]) or 0.01

            # Вероятность роста через нормальное распределение
            prob_up = 1 - norm.cdf(0, loc=predicted_return, scale=se)
            prob_up = np.clip(prob_up, 0.001, 0.999)  # не даём крайних 0 и 1

            direction = 1 if predicted_return > 0 else 0

            return {
                'direction': int(direction),
                'prob_up': float(prob_up),
                'predicted_return': float(predicted_return)
            }
        except Exception:
            return {'direction': 1, 'prob_up': 0.5}