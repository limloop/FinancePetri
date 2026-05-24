class BaseTest:
    name = "BaseTest"
    default_n_runs = 10

    def __init__(self, n_runs=None, params=None):
        self.n_runs = n_runs or self.default_n_runs
        self.params = params or {}

    def prepare_train_window(self, data_bus, run_id, symbol=None):
        raise NotImplementedError

    def prepare_test_window(self, data_bus, run_id, symbol=None):
        raise NotImplementedError

    def get_ground_truth(self, test_slice):
        raise NotImplementedError

    def evaluate(self, prediction: dict, ground_truth: dict) -> dict:
        raise NotImplementedError