import numpy as np
import warnings
from arch import arch_model
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class GARCHAutoModel(BaseModel):
    """
    GARCH с автоматическим подбором порядка (p, q) по AIC.
    Перебирает p ∈ {1,2}, q ∈ {1,2} и выбирает лучшую модель.
    """
    name = "GARCHAutoModel"
    required_fields = ['returns']

    def __init__(self, seed=None, max_p=2, max_q=2, mean='constant'):
        self.max_p = max_p
        self.max_q = max_q
        self.mean = mean
        self.predicted_vol = 0.01
        self.best_order = (1, 1)
        self.fitted = False

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean = returns[~np.isnan(returns)]

        if len(clean) < 30:
            self.predicted_vol = float(np.nanstd(clean)) if len(clean) > 0 else 0.01
            self.fitted = False
            return

        best_aic = float('inf')
        best_result = None
        best_order = (1, 1)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            scaled_returns = clean * 100

            for p in range(1, self.max_p + 1):
                for q in range(1, self.max_q + 1):
                    try:
                        model = arch_model(
                            scaled_returns,
                            vol='GARCH',
                            p=p,
                            q=q,
                            mean=self.mean,
                            dist='normal'
                        )
                        result = model.fit(disp='off', show_warning=False)

                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_result = result
                            best_order = (p, q)
                    except Exception:
                        continue

        self.best_order = best_order

        if best_result is not None:
            try:
                forecast = best_result.forecast(horizon=1)
                variance_forecast = forecast.variance.values[-1, 0]
                self.predicted_vol = float(np.sqrt(variance_forecast)) / 100
                self.fitted = True
            except Exception:
                self.predicted_vol = float(np.nanstd(clean))
                self.fitted = False
        else:
            self.predicted_vol = float(np.nanstd(clean))
            self.fitted = False

        if np.isnan(self.predicted_vol) or self.predicted_vol <= 0:
            self.predicted_vol = float(np.nanstd(clean))

        self.predicted_vol = max(self.predicted_vol, 1e-6)

    def predict(self, data: dict) -> dict:
        return {
            'predicted_volatility': float(self.predicted_vol),
            'annualized_volatility': float(self.predicted_vol * np.sqrt(252))
        }