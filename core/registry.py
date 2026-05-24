class Registry:
    def __init__(self):
        self._models = {}
        self._tests = {}

    def register_model(self, model_class):
        """Декоратор для регистрации модели."""
        name = model_class.name
        self._models[name] = model_class
        return model_class

    def register_test(self, test_class):
        """Декоратор для регистрации теста."""
        name = test_class.name
        self._tests[name] = test_class
        return test_class

    def get_model(self, name):
        return self._models.get(name)

    def get_test(self, name):
        return self._tests.get(name)

    @property
    def available_models(self):
        return list(self._models.keys())

    @property
    def available_tests(self):
        return list(self._tests.keys())


# Глобальный экземпляр реестра
registry = Registry()