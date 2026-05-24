import numpy as np
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class RandomModel(BaseModel):
    name = "RandomModel"
    required_fields = ['returns']

    def fit(self, data: dict) -> None:
        # Случайной модели нечему учиться
        pass

    def predict(self, data: dict) -> dict:
        # С вероятностью 0.5 предсказываем рост, 0.5 — падение
        direction = np.random.choice([0, 1])
        prob_up = 0.5  # вероятности нет, просто константа для совместимости

        return {
            'direction': int(direction),
            'prob_up': float(prob_up)
        }