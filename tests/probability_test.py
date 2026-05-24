import numpy as np
from tests.base_test import BaseTest
from core.registry import registry


@registry.register_test
class ProbabilityTest(BaseTest):
    name = "ProbabilityTest"
    default_n_runs = 50

    def __init__(self, n_runs=None, params=None):
        super().__init__(n_runs, params)
        self.train_days = self.params.get('train_days', 200)
        self.test_days = self.params.get('test_days', 5)
        self.horizon = self.params.get('horizon', 1)
        self.n_bins = self.params.get('n_bins', 10)

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
                actual_direction = 1 if ret > 0 else 0
            else:
                actual_direction = None
        else:
            actual_direction = None
        return {'actual_direction': actual_direction}

    def evaluate(self, prediction: dict, ground_truth: dict) -> dict:
        actual = ground_truth.get('actual_direction')
        prob_up = prediction.get('prob_up')

        if actual is None or prob_up is None:
            return {'valid': False}

        brier = (prob_up - actual) ** 2

        eps = 1e-15
        p_clipped = np.clip(prob_up, eps, 1 - eps)
        log_loss = -(actual * np.log(p_clipped) + (1 - actual) * np.log(1 - p_clipped))

        return {
            'valid': True,
            'brier_score': float(brier),
            'log_loss': float(log_loss),
            'prob_up': float(prob_up),
            'actual': int(actual),
        }