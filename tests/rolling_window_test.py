import numpy as np
from tests.base_test import BaseTest
from core.registry import registry


@registry.register_test
class RollingWindowTest(BaseTest):
    name = "RollingWindowTest"
    default_n_runs = 100

    def __init__(self, n_runs=None, params=None):
        super().__init__(n_runs, params)
        self.train_days = self.params.get('train_days', 100)
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
            ret = returns[horizon]
            if not np.isnan(ret):
                direction = 1 if ret > 0 else 0
            else:
                direction = None
        else:
            direction = None
        return {'direction': direction}

    def evaluate(self, prediction: dict, ground_truth: dict) -> dict:
        actual = ground_truth.get('direction')
        pred = prediction.get('direction')

        if actual is None or pred is None:
            return {'valid': False}

        acc = 1.0 if pred == actual else 0.0

        return {
            'valid': True,
            'accuracy': acc,
            'window_size': self.train_days,
            'actual': int(actual),
            'predicted': int(pred),
        }