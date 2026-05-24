from tests.direction_test import DirectionTest
from core.registry import registry


@registry.register_test
class DirectionTest50(DirectionTest):
    name = "DirectionTest50"
    default_n_runs = 200

    def __init__(self, n_runs=None, params=None):
        super().__init__(n_runs=n_runs, params=params)
        self.train_days = 50


@registry.register_test
class DirectionTest100(DirectionTest):
    name = "DirectionTest100"
    default_n_runs = 200

    def __init__(self, n_runs=None, params=None):
        super().__init__(n_runs=n_runs, params=params)
        self.train_days = 100


@registry.register_test
class DirectionTest200(DirectionTest):
    name = "DirectionTest200"
    default_n_runs = 200

    def __init__(self, n_runs=None, params=None):
        super().__init__(n_runs=n_runs, params=params)
        self.train_days = 200