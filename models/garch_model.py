import numpy as np
import warnings
from arch import arch_model
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class GARCHModel(BaseModel):
    """
    GARCH(1,1) модель для прогноза волатильности.
    Предсказывает стандартное отклонение доходности на следующий день.
    """
    name = "GARCHModel"
    required_fields = ['returns']

    def __init__(self, seed=None, p=1, q=1, mean='constant'):
        self.p = p
        self.q = q
        self.mean = mean
        self.predicted_vol = 0.01
        self.fitted = False

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean = returns[~np.isnan(returns)]

        if len(clean) < 30:
            self.predicted_vol = float(np.nanstd(clean)) if len(clean) > 0 else 0.01
            self.fitted = False
            return

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # Масштабируем доходности в проценты для улучшения сходимости
                scaled_returns = clean * 100

                model = arch_model(
                    scaled_returns,
                    vol='GARCH',
                    p=self.p,
                    q=self.q,
                    mean=self.mean,
                    dist='normal'
                )
                result = model.fit(disp='off', show_warning=False)

                # Прогноз на 1 шаг вперёд
                forecast = result.forecast(horizon=1)
                variance_forecast = forecast.variance.values[-1, 0]

                # Обратно масштабируем
                self.predicted_vol = float(np.sqrt(variance_forecast)) / 100

                if np.isnan(self.predicted_vol) or self.predicted_vol <= 0:
                    self.predicted_vol = float(np.nanstd(clean))

                self.fitted = True

        except Exception:
            # Fallback: историческая волатильность
            self.predicted_vol = float(np.nanstd(clean)) if len(clean) > 0 else 0.01
            self.fitted = False

        self.predicted_vol = max(self.predicted_vol, 1e-6)

    def predict(self, data: dict) -> dict:
        return {
            'predicted_volatility': float(self.predicted_vol),
            'annualized_volatility': float(self.predicted_vol * np.sqrt(252))
        }