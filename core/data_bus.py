import yfinance as yf
import pandas as pd
import numpy as np
from core.market_regime import MarketRegimeDetector


class DataBus:
    """Загружает данные, предоставляет срезы, обогащает информацией о режиме."""

    def __init__(self, symbols, start_date, end_date):
        if isinstance(symbols, str):
            symbols = [symbols]
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.detector = MarketRegimeDetector()

    def download(self):
        print(f"Загружаем данные для {self.symbols}...")
        raw = yf.download(self.symbols, start=self.start_date, end=self.end_date, progress=False)

        for symbol in self.symbols:
            if len(self.symbols) == 1:
                close = raw['Close']
            else:
                close = raw['Close'][symbol]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = close.dropna()
            returns = np.log(close / close.shift(1))

            self.data[symbol] = {
                'close': close.values,
                'returns': returns.values,
                'dates': close.index,
            }
            print(f"  {symbol}: {len(close)} дней")

        return self

    def get_slice(self, symbol, start_idx, end_idx):
        if symbol not in self.data:
            raise KeyError(f"Символ {symbol} не загружен. Доступны: {list(self.data.keys())}")

        d = self.data[symbol]
        sliced = {}
        for key, val in d.items():
            if isinstance(val, np.ndarray):
                sliced[key] = val[start_idx:end_idx]
            else:
                sliced[key] = val
        return sliced

    def enrich_with_regime(self, data_slice: dict) -> dict:
        """
        Добавляет информацию о режиме рынка в словарь данных.
        Модели могут использовать data['regime'], data['regime_confidence'] и т.д.
        """
        returns = data_slice.get('returns', np.array([]))
        regime_info = self.detector.detect(returns)

        # Добавляем плоские ключи для удобства
        enriched = dict(data_slice)
        enriched['regime'] = regime_info['regime']
        enriched['regime_confidence'] = regime_info['confidence']
        enriched['regime_metrics'] = regime_info['metrics']

        return enriched

    def get_symbols(self):
        return list(self.data.keys())