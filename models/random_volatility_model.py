import numpy as np
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class RandomVolatilityModel(BaseModel):
    """
    Случайный прогноз волатильности — baseline.
    Выдаёт историческую волатильность, умноженную на случайный шум (0.5x–1.5x).
    """
    name = "RandomVolatilityModel"
    required_fields = ['returns']

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.predicted_vol = 0.01

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean = returns[~np.isnan(returns)]

        if len(clean) > 0:
            self.predicted_vol = float(np.nanstd(clean))
        else:
            self.predicted_vol = 0.01

    def predict(self, data: dict) -> dict:
        # Случайный множитель от 0.5 до 1.5
        noise = self.rng.uniform(0.5, 1.5)
        random_vol = self.predicted_vol * noise

        return {
            'predicted_volatility': float(random_vol),
            'annualized_volatility': float(random_vol * np.sqrt(252))
        }