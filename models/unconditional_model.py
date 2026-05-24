import numpy as np
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class UnconditionalModel(BaseModel):
    """
    Безусловная модель: предсказывает prob_up = доля растущих дней в обучении.
    Это baseline, который учитывает тренд, но не автокорреляцию.
    """
    name = "UnconditionalModel"
    required_fields = ['returns']

    def __init__(self, seed=None):
        self.prob_up = 0.5

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean = returns[~np.isnan(returns)]

        if len(clean) == 0:
            self.prob_up = 0.5
            return

        # Доля положительных доходностей
        self.prob_up = float((clean > 0).mean())

        # Защита от крайних значений
        self.prob_up = np.clip(self.prob_up, 0.001, 0.999)

    def predict(self, data: dict) -> dict:
        direction = 1 if self.prob_up >= 0.5 else 0

        return {
            'direction': int(direction),
            'prob_up': float(self.prob_up)
        }