import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm


class Reporter:
    def __init__(self, show_progress=True, results_dir='results'):
        self.show_progress = show_progress
        self.results_dir = results_dir
        self.results = []
        os.makedirs(results_dir, exist_ok=True)

    def record(self, model_name, test_name, run_id, symbol, metrics):
        self.results.append({
            'symbol': symbol,
            'model': model_name,
            'test': test_name,
            'run': run_id,
            **metrics
        })

    def get_progress_bar(self, total):
        if self.show_progress:
            return tqdm(total=total, desc="FinancePetri", unit="run")
        else:
            class DummyProgress:
                def update(self, n=1): pass
                def set_postfix(self, *a, **k): pass
                def close(self): pass
            return DummyProgress(total)

    def summary(self):
        if not self.results:
            print("\nНет результатов.")
            return None

        df = pd.DataFrame(self.results)

        # Фильтруем только валидные
        if 'valid' in df.columns:
            df_valid = df[df['valid'] == True].copy()
        else:
            df_valid = df

        if df_valid.empty:
            print("\nНет валидных результатов.")
            return None

        # === Сохранение ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Полный файл с сырыми данными (все прогоны)
        full_path = os.path.join(self.results_dir, f"run_{timestamp}.csv")
        df.to_csv(full_path, index=False)

        # Сводный файл (агрегированный)
        agg_cols = ['symbol', 'model', 'test']
        metric_cols = self._get_metric_columns(df_valid)
        agg_dict = {col: ['mean', 'std'] for col in metric_cols}
        agg_dict['run'] = 'count'

        summary_df = df_valid.groupby(agg_cols).agg(agg_dict).reset_index()
        summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns]
        summary_df = summary_df.rename(columns={'run_count': 'n_runs'})

        summary_path = os.path.join(self.results_dir, f"summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)

        # Дописываем в историю
        history_path = os.path.join(self.results_dir, "history.csv")
        df['timestamp'] = timestamp
        if os.path.exists(history_path):
            df.to_csv(history_path, mode='a', header=False, index=False)
        else:
            df.to_csv(history_path, index=False)

        # === Вывод в консоль ===
        print("\n" + "=" * 80)
        print("  FinancePetri — Сводка результатов")
        print("=" * 80)

        tests = df_valid['test'].unique()
        for test_name in sorted(tests):
            test_df = df_valid[df_valid['test'] == test_name]
            self._print_test_block(test_name, test_df)

        print(f"\n  Полные результаты: {full_path}")
        print(f"  Сводка: {summary_path}")
        print(f"  История: {history_path}")
        print("=" * 80)

        return df_valid

    def _print_test_block(self, test_name, test_df):
        print(f"\n  ▸ {test_name}")

        symbols = sorted(test_df['symbol'].unique())
        metric_cols = self._get_metric_columns(test_df)

        if not metric_cols:
            print("    (нет метрик)")
            return

        for sym in symbols:
            sym_df = test_df[test_df['symbol'] == sym]
            print(f"\n    Символ: {sym}")

            models = sorted(sym_df['model'].unique())
            rows = []
            for model in models:
                model_df = sym_df[sym_df['model'] == model]
                row = {'Модель': model, 'Прогонов': len(model_df)}
                for col in metric_cols:
                    values = model_df[col].dropna()
                    if len(values) > 0:
                        row[col] = f"{values.mean():.4f} ± {values.std():.4f}"
                    else:
                        row[col] = "—"
                rows.append(row)

            result_df = pd.DataFrame(rows)
            cols_order = ['Модель', 'Прогонов'] + [c for c in metric_cols if c not in ['Модель', 'Прогонов']]
            result_df = result_df[[c for c in cols_order if c in result_df.columns]]

            print(result_df.to_string(index=False))

    def _get_metric_columns(self, df):
        candidates = [
            'accuracy', 'brier_score', 'log_loss',
            'always_up_accuracy', 'always_down_accuracy',
            'mse', 'mae', 'bias', 'hit_rate',
        ]
        return [c for c in candidates if c in df.columns and df[c].notna().any()]