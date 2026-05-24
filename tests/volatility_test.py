import numpy as np
from tests.base_test import BaseTest
from core.registry import registry


@registry.register_test
class VolatilityTest(BaseTest):
    """
    Оценивает качество прогноза волатильности.

    Метрики:
      - MSE: средний квадрат ошибки (predicted_vol - |actual_return|)^2
      - MAE: средняя абсолютная ошибка
      - Bias: среднее отклонение (переоценка/недооценка)
      - Hit Rate: доля дней, когда |return| <= 2 * predicted_vol (примерно 95% доверительный интервал)
    """
    name = "VolatilityTest"
    default_n_runs = 100

    def __init__(self, n_runs=None, params=None):
        super().__init__(n_runs, params)
        self.train_days = self.params.get('train_days', 200)
        self.test_days = self.params.get('test_days', 5)
        self.horizon = self.params.get('horizon', 1)

    def prepare_train_window(self, data_bus, run_id, symbol=None):
        start = run_id
        end = start + self.train_days
        sym = symbol or data_bus.get_symbols()[0]
        return data_bus.get_slice(sym, start, end)

    def prepare_test_window(self, data_bus, run_id, symbol=None):
        start = run_id + self.train_days
        end = start + self.test_days + self.horizon
        sym = symbol or data_bus.get_symbols()[0]
        return data_bus.get_slice(sym, start, end)

    def get_ground_truth(self, test_slice):
        returns = test_slice['returns']
        horizon = self.horizon

        if len(returns) > horizon:
            actual_return = returns[horizon]
            if not np.isnan(actual_return):
                actual_vol = abs(actual_return)
            else:
                actual_vol = None
        else:
            actual_vol = None

        return {
            'actual_volatility': actual_vol,
            'actual_return': actual_return if len(returns) > horizon else None
        }

    def evaluate(self, prediction: dict, ground_truth: dict) -> dict:
        actual_vol = ground_truth.get('actual_volatility')
        predicted_vol = prediction.get('predicted_volatility')

        if actual_vol is None or predicted_vol is None or predicted_vol <= 0:
            return {'valid': False}

        # MSE
        mse = (predicted_vol - actual_vol) ** 2

        # MAE
        mae = abs(predicted_vol - actual_vol)

        # Bias: положительный = переоценка, отрицательный = недооценка
        bias = predicted_vol - actual_vol

        # Hit Rate: попадает ли |return| в 2-сигма интервал
        actual_return = ground_truth.get('actual_return')
        hit = 1 if (actual_return is not None and abs(actual_return) <= 2 * predicted_vol) else 0

        return {
            'valid': True,
            'mse': float(mse),
            'mae': float(mae),
            'bias': float(bias),
            'hit_rate': hit,
            'predicted_vol': float(predicted_vol),
            'actual_vol': float(actual_vol),
        }