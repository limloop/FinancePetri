# FinancePetri 🧫

Лаборатория для экспериментов с алгоритмическими моделями финансовых рынков.

**FinancePetri** — это фреймворк для честного тестирования торговых алгоритмов. Название отсылает к чашке Петри: вы помещаете алгоритм в питательную среду исторических данных и наблюдаете, «вырастет» ли из него прибыль.

## 🎯 Философия

- **Честный бэктестинг.** Скользящее окно (walk-forward validation) — никакого look-ahead bias.
- **Доказательная база.** Каждая модель сравнивается с baseline (RandomModel) и наивными стратегиями.
- **Модульность.** Добавить новую модель или тест можно за 5 минут.

---

## 🏗 Архитектура

```
finance_petri/
├── main.py                          # Точка входа
├── config.yaml                      # Управление моделями и тестами
├── README.md                        # Этот файл
├── core/
│   ├── engine.py                    # Главный цикл: данные → модели → тесты
│   ├── data_bus.py                  # Загрузка данных (yfinance)
│   ├── registry.py                  # Реестр моделей и тестов через декораторы
│   ├── imports.py                   # Умный импорт (только enabled: true)
│   └── market_regime.py             # Детектор рыночных режимов
├── models/
│   ├── base_model.py                # Интерфейс модели (fit/predict)
│   ├── random_model.py              # Случайное гадание (baseline)
│   ├── random_volatility_model.py   # Случайная волатильность (baseline)
│   ├── arima_model.py               # ARIMA(2,0,1)
│   ├── arima_auto_model.py          # ARIMA с авто-подбором порядка
│   ├── unconditional_model.py       # Безусловная вероятность роста
│   ├── unconditional_model_v2.py    # EMA-взвешенная вероятность
│   ├── unconditional_model_v3.py    # Режимное снижение уверенности
│   ├── unconditional_ensemble.py    # Ансамбль трёх окон
│   ├── unconditional_regime_selector.py  # Адаптивный выбор стратегии
│   ├── logistic_model.py            # Логистическая регрессия (6 признаков)
│   ├── logistic_model_v2.py         # Логистическая регрессия (12 признаков)
│   ├── historical_volatility_model.py    # Средняя волатильность
│   ├── historical_volatility_model_v2.py # Адаптивное окно
│   ├── exponential_volatility_model.py   # EWMA волатильность
│   ├── garch_model.py               # GARCH(1,1)
│   ├── garch_auto_model.py          # GARCH с авто-подбором порядка
│   └── hybrid_ensemble_model.py     # Гибридный ансамбль
├── tests/
│   ├── base_test.py                 # Интерфейс теста
│   ├── direction_test.py            # Accuracy направления
│   ├── direction_window_tests.py    # DirectionTest с окнами 50/100/200
│   ├── probability_test.py          # Brier score и log loss
│   ├── benchmark_test.py            # Сравнение с always_up/always_down
│   ├── rolling_window_test.py       # Стабильность на скользящем окне
│   └── volatility_test.py           # MSE, MAE, Bias, Hit Rate
├── metrics/
│   └── reporter.py                  # Вывод в консоль + CSV
└── results/                         # История всех запусков
```

---

## 🚀 Быстрый старт

### Установка

```bash
pip install yfinance pandas numpy tqdm pyyaml scipy statsmodels scikit-learn arch
```

### Запуск

```bash
python main.py
```

Все настройки — в `config.yaml`. Выберите активы, включите нужные модели и тесты.

---

## 📊 Модели

### Направление (Direction)

| Модель | Принцип | Статус |
|---|---|---|
| `RandomModel` | Честная монетка (baseline) | ✅ Для оценки других |
| `UnconditionalModel` | Средняя доля ростов в окне | 🏆 Чемпион трендов |
| `UnconditionalModelV2` | EMA-взвешенная доля ростов | ❌ Не оправдала |
| `UnconditionalModelV3` | Режимное снижение уверенности | 🏆 Чемпион боковиков |
| `UnconditionalEnsemble` | Ансамбль окон 20/60/100 дней | ✅ Стабильный универсал |
| `UnconditionalRegimeSelector` | Адаптивный выбор стратегии | ✅ Хорош на трендах |
| `LogisticModel` | Логистическая регрессия (6 признаков) | ❌ Выродилась |
| `LogisticModelV2` | Логистическая регрессия (12 признаков) | ❌ Выродилась |
| `ArimaModel` | ARIMA(2,0,1) | ❌ Не лучше среднего |
| `ArimaAutoModel` | ARIMA с авто-подбором порядка | ❌ Не лучше среднего |
| `HybridEnsembleModel` | Гибридный ансамбль (3 направления + 2 волатильности) | ✅ Стабильный |

### Волатильность (Volatility)

| Модель | Принцип | Статус |
|---|---|---|
| `RandomVolatilityModel` | Случайный шум вокруг средней (baseline) | ✅ Для оценки других |
| `HistoricalVolatilityModel` | Среднее std за 20 дней | ✅ Надёжный baseline |
| `HistoricalVolatilityModelV2` | Адаптивное окно (10/20/60 дней) | 🏆 Чемпион |
| `ExponentialVolatilityModel` | EWMA с alpha=0.94 | 🥈 Быстрая адаптация |
| `GARCHModel` | GARCH(1,1) | ❌ Не лучше Historical |
| `GARCHAutoModel` | GARCH с авто-подбором порядка | ❌ Не лучше Historical |
| `HybridEnsembleModel` | Гибридный ансамбль (3 направления + 2 волатильности) | ✅ Стабильный |

---

## 🧪 Тесты

| Тест | Метрики | Что измеряет |
|---|---|---|
| `DirectionTest` | Accuracy | Доля угаданных направлений |
| `DirectionTest50/100/200` | Accuracy | То же с окнами 50/100/200 дней для одновременного использования |
| `ProbabilityTest` | Brier Score, Log Loss | Качество вероятностных прогнозов |
| `BenchmarkTest` | Accuracy vs AlwaysUp/AlwaysDown | Сравнение с наивными стратегиями |
| `RollingWindowTest` | Accuracy | Стабильность на скользящем окне |
| `VolatilityTest` | MSE, MAE, Bias, Hit Rate | Точность прогноза волатильности |

---

## 📈 Демонстрационные результаты

### Лучшие Brier Score (направление)

| Модель | Актив | Brier Score | Окно |
|---|---|---|---|
| `UnconditionalModel` | ACHR | **0.2336** | 100 дней |
| `HybridEnsembleModel` | TSLA | **0.2397** | 400 дней |
| `UnconditionalModel` | TSLA | **0.2399** | 400 дней |
| `UnconditionalEnsemble` | TSLA | **0.2405** | 400 дней |
| `UnconditionalRegimeSelector` | TSLA | **0.2417** | 100 дней |

### Лучшие MAE (волатильность)

| Модель | Актив | MAE | Окно |
|---|---|---|---|
| `HistoricalVolatilityModelV2` | NKE | **0.0073** | 400 дней |
| `HistoricalVolatilityModel` | NKE | **0.0078** | 400 дней |
| `HistoricalVolatilityModel` | AAPL | **0.0080** | 400 дней |
| `HistoricalVolatilityModelV2` | AAPL | **0.0081** | 400 дней |
| `ExponentialVolatilityModel` | AAPL | **0.0082** | 400 дней |

## 🛠 Конфигурация

### Добавление новой модели

1. Создать `models/my_model.py`:
```python
from models.base_model import BaseModel
from core.registry import registry

@registry.register_model
class MyModel(BaseModel):
    name = "MyModel"
    required_fields = ['returns']
    
    def fit(self, data): ...
    def predict(self, data): ...
```

2. Добавить в `core/imports.py`:
```python
'MyModel': 'models.my_model',
```

3. Включить в `config.yaml`:
```yaml
models:
  - name: "MyModel"
    enabled: true
```

### Добавление нового теста

Аналогично: создать `tests/my_test.py`, наследовать `BaseTest`, зарегистрировать в `imports.py`.

---

## 📚 Словарь терминов

- **Brier Score** — средний квадрат ошибки вероятностей (0 = идеально, 0.25 = монетка)
- **Log Loss** — логарифмическая функция потерь (−ln(p) для правильного класса)
- **MAE** — средняя абсолютная ошибка прогноза волатильности
- **Hit Rate** — доля дней, когда |доходность| ≤ 2 × прогноз волатильности (должно быть ~0.95)
- **Walk-forward validation** — обучение на прошлом, тест на будущем, без подглядывания