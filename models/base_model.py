class BaseModel:
    name = "BaseModel"        # переопределить в наследнике
    required_fields = []      # список ключей, которые модель берёт из словаря данных

    def fit(self, data: dict) -> None:
        raise NotImplementedError

    def predict(self, data: dict) -> dict:
        """
        Возвращает словарь с предсказанием.
        Например: {'direction': 1, 'prob_up': 0.62}
        """
        raise NotImplementedError