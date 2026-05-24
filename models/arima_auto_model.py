import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class ArimaAutoModel(BaseModel):
    """
    ARIMA с автоматическим подбором порядка (p, d, q) по AIC.
    Перебирает p в [0..max_p], q в [0..max_q], d=0 (доходности стационарны).
    """
    name = "ArimaAutoModel"
    required_fields = ['returns']

    def __init__(self, max_p=3, max_q=3, suppress_warnings=True):
        self.max_p = max_p
        self.max_q = max_q
        self.suppress_warnings = suppress_warnings
        self.fitted_model = None
        self.best_order = None

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean = returns[~np.isnan(returns)]

        if len(clean) < 30:
            self.fitted_model = None
            return

        best_aic = np.inf
        best_result = None
        best_order = (0, 0, 0)

        with warnings.catch_warnings():
            if self.suppress_warnings:
                warnings.filterwarnings('ignore')

            for p in range(self.max_p + 1):
                for q in range(self.max_q + 1):
                    if p == 0 and q == 0:
                        continue  # ARMA(0,0) — просто белый шум, неинтересно
                    try:
                        model = ARIMA(
                            clean,
                            order=(p, 0, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        result = model.fit(method_kwargs={'maxiter': 100})
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_result = result
                            best_order = (p, 0, q)
                    except Exception:
                        continue

        self.fitted_model = best_result
        self.best_order = best_order

    def predict(self, data: dict) -> dict:
        if self.fitted_model is None:
            return {'direction': 1, 'prob_up': 0.5}

        try:
            forecast_result = self.fitted_model.get_forecast(steps=1)
            predicted_return = forecast_result.predicted_mean[0]
            se = forecast_result.se_mean[0]

            if se <= 0 or np.isnan(se):
                clean = data['returns'][~np.isnan(data['returns'])]
                se = np.std(clean) if len(clean) > 0 else 0.01

            prob_up = 1 - norm.cdf(0, loc=predicted_return, scale=se)
            prob_up = np.clip(prob_up, 0.001, 0.999)
            direction = 1 if predicted_return > 0 else 0

            return {
                'direction': int(direction),
                'prob_up': float(prob_up),
                'predicted_return': float(predicted_return),
                'order': self.best_order,
            }
        except Exception:
            return {'direction': 1, 'prob_up': 0.5}