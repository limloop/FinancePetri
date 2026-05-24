"""
Умный импорт моделей и тестов.
Читает config.yaml и импортирует только включённые (enabled: true) компоненты.
Если config не найден или не читается — импортирует всё для обратной совместимости.
"""

import yaml
import os
import importlib

# Все известные модели и тесты с путями для импорта
AVAILABLE_MODELS = {
    'RandomModel': 'models.random_model',
    'RandomVolatilityModel': 'models.random_volatility_model',

    'ArimaModel': 'models.arima_model',
    'ArimaAutoModel': 'models.arima_auto_model',

    'UnconditionalModel': 'models.unconditional_model',
    'UnconditionalModelV2': 'models.unconditional_model_v2',
    'UnconditionalModelV3': 'models.unconditional_model_v3',
    'UnconditionalEnsemble': 'models.unconditional_ensemble',
    'UnconditionalRegimeSelector': 'models.unconditional_regime_selector',

    'LogisticModel': 'models.logistic_model',
    'LogisticModelV2': 'models.logistic_model_v2',

    'HistoricalVolatilityModel': 'models.historical_volatility_model',
    'HistoricalVolatilityModelV2': 'models.historical_volatility_model_v2',
    'ExponentialVolatilityModel': 'models.exponential_volatility_model',

    'GARCHModel': 'models.garch_model',
    'GARCHAutoModel': 'models.garch_auto_model',

    'HybridEnsembleModel': 'models.hybrid_ensemble_model',
}

AVAILABLE_TESTS = {
    'DirectionTest': 'tests.direction_test',
    'DirectionTest50': 'tests.direction_window_tests',
    'DirectionTest100': 'tests.direction_window_tests',
    'DirectionTest200': 'tests.direction_window_tests',
    
    'ProbabilityTest': 'tests.probability_test',
    'BenchmarkTest': 'tests.benchmark_test',
    'RollingWindowTest': 'tests.rolling_window_test',

    'VolatilityTest': 'tests.volatility_test',
}

# Обратный словарь: путь модуля → список классов в нём
MODULE_TO_CLASSES = {}
for class_name, module_path in {**AVAILABLE_MODELS, **AVAILABLE_TESTS}.items():
    MODULE_TO_CLASSES.setdefault(module_path, []).append(class_name)


def load_config_names(config_path='config.yaml'):
    """Читает config.yaml и возвращает множества включённых моделей и тестов."""
    enabled_models = set()
    enabled_tests = set()

    if not os.path.exists(config_path):
        print(f"config.yaml не найден, импортируем всё.")
        return set(AVAILABLE_MODELS.keys()), set(AVAILABLE_TESTS.keys())

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        for m in config.get('models', []):
            if m.get('enabled', True):
                enabled_models.add(m['name'])

        for t in config.get('tests', []):
            if t.get('enabled', True):
                enabled_tests.add(t['name'])

    except Exception as e:
        print(f"Ошибка чтения config.yaml: {e}, импортируем всё.")
        return set(AVAILABLE_MODELS.keys()), set(AVAILABLE_TESTS.keys())

    return enabled_models, enabled_tests


def smart_import():
    """
    Импортирует только те модули, которые содержат включённые модели/тесты.
    Если модуль содержит несколько классов (например, direction_window_tests),
    он импортируется, если включён хотя бы один из них.
    """
    enabled_models, enabled_tests = load_config_names()

    # Собираем, какие модули нужно импортировать
    modules_to_import = set()

    for name in enabled_models:
        if name in AVAILABLE_MODELS:
            modules_to_import.add(AVAILABLE_MODELS[name])
        else:
            print(f"Модель '{name}' из config.yaml не найдена в AVAILABLE_MODELS.")

    for name in enabled_tests:
        if name in AVAILABLE_TESTS:
            modules_to_import.add(AVAILABLE_TESTS[name])
        else:
            print(f"Тест '{name}' из config.yaml не найден в AVAILABLE_TESTS.")

    # Импортируем
    for module_path in sorted(modules_to_import):
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            print(f"Ошибка импорта {module_path}: {e}")

    print(f"Импортированы модели: {sorted(enabled_models & set(AVAILABLE_MODELS.keys()))}")
    print(f"Импортированы тесты: {sorted(enabled_tests & set(AVAILABLE_TESTS.keys()))}")


# Автоматически запускаем при импорте этого модуля
smart_import()