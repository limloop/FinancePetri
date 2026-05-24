import numpy as np
from models.base_model import BaseModel
from core.registry import registry


@registry.register_model
class UnconditionalModelV3(BaseModel):
    """
    Режимная безусловная модель.
    Определяет режим рынка (тренд/боковик) и адаптирует прогноз.
    В боковике снижает уверенность до 0.5.
    """
    name = "UnconditionalModelV3"
    required_fields = ['returns']

    def __init__(self, seed=None, trend_threshold=0.55, vol_threshold=1.5):
        """
        trend_threshold: если доля ростов > 0.55 или < 0.45 — считаем трендом
        vol_threshold: если текущая волатильность > средняя * vol_threshold — неопределённость
        """
        self.trend_threshold = trend_threshold
        self.vol_threshold = vol_threshold
        self.prob_up = 0.5

    def fit(self, data: dict) -> None:
        returns = data['returns']
        clean = returns[~np.isnan(returns)]

        if len(clean) < 20:
            self.prob_up = 0.5
            return

        # Базовая вероятность
        prob_up_raw = float((clean > 0).mean())

        # Сила тренда: насколько prob_up отклоняется от 0.5
        trend_strength = abs(prob_up_raw - 0.5) * 2  # 0 = нет тренда, 1 = чистый тренд

        # Волатильность
        current_vol = np.nanstd(clean[-10:]) if len(clean) >= 10 else np.nanstd(clean)
        historical_vol = np.nanstd(clean)
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0

        # Если волатильность аномально высокая — снижаем уверенность
        confidence = 1.0
        if vol_ratio > self.vol_threshold:
            confidence *= 0.5  # высокая волатильность → меньше уверенности

        # Если тренд слабый — снижаем уверенность
        if trend_strength < 0.2:  # доля ростов между 0.4 и 0.6
            confidence *= 0.5

        # Стягиваем к 0.5 пропорционально неуверенности
        self.prob_up = 0.5 + (prob_up_raw - 0.5) * confidence
        self.prob_up = np.clip(self.prob_up, 0.001, 0.999)

    def predict(self, data: dict) -> dict:
        direction = 1 if self.prob_up >= 0.5 else 0

        return {
            'direction': int(direction),
            'prob_up': float(self.prob_up)
        }