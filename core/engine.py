import yaml
from core.registry import registry
from core.data_bus import DataBus
from metrics.reporter import Reporter


class Engine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.reporter = Reporter(
            show_progress=self.config['output'].get('show_progress', True),
            results_dir=self.config['output'].get('results_dir', 'results')
        )

    def run(self):
        # 1. Данные
        global_cfg = self.config['global']
        data_bus = DataBus(
            symbols=global_cfg['symbols'],
            start_date=global_cfg['start_date'],
            end_date=global_cfg['end_date']
        )
        data_bus.download()
        symbols = data_bus.get_symbols()

        # 2. Конфиги моделей
        enabled_model_configs = []
        for m_cfg in self.config.get('models', []):
            if not m_cfg.get('enabled', True):
                continue
            model_cls = registry.get_model(m_cfg['name'])
            if model_cls is None:
                print(f"Модель '{m_cfg['name']}' не найдена в реестре.")
                continue
            enabled_model_configs.append((m_cfg['name'], model_cls, m_cfg.get('params', {})))

        # 3. Конфиги тестов
        enabled_test_configs = []
        for t_cfg in self.config.get('tests', []):
            if not t_cfg.get('enabled', True):
                continue
            test_cls = registry.get_test(t_cfg['name'])
            if test_cls is None:
                print(f"Тест '{t_cfg['name']}' не найден в реестре.")
                continue
            enabled_test_configs.append((t_cfg['name'], test_cls, t_cfg.get('n_runs'), t_cfg.get('params', {})))

        # 4. Общее число прогонов
        total_runs = sum(n_runs for _, _, n_runs, _ in enabled_test_configs) * len(enabled_model_configs) * len(symbols)
        if total_runs == 0:
            print("Нет включённых моделей или тестов. Проверьте config.yaml")
            return

        progress = self.reporter.get_progress_bar(total_runs)

        # 5. Главный цикл
        for symbol in symbols:
            for model_name, model_cls, model_params in enabled_model_configs:
                for test_name, test_cls, n_runs, test_params in enabled_test_configs:
                    test = test_cls(n_runs=n_runs, params=test_params)

                    for run_id in range(test.n_runs):
                        train_slice_raw = test.prepare_train_window(data_bus, run_id, symbol)
                        test_slice_raw = test.prepare_test_window(data_bus, run_id, symbol)

                        returns = train_slice_raw.get('returns')
                        if returns is None or len(returns) < 10:
                            self.reporter.record(model_name, test_name, run_id, symbol, {'valid': False})
                            progress.update(1)
                            continue

                        # Обогащаем данные информацией о режиме
                        train_slice = data_bus.enrich_with_regime(train_slice_raw)
                        test_slice = data_bus.enrich_with_regime(test_slice_raw)

                        try:
                            model = model_cls(seed=run_id)
                        except TypeError:
                            model = model_cls()

                        model.fit(train_slice)
                        prediction = model.predict(test_slice)
                        ground_truth = test.get_ground_truth(test_slice)
                        metrics = test.evaluate(prediction, ground_truth)

                        self.reporter.record(model_name, test_name, run_id, symbol, metrics)

                        if self.reporter.show_progress:
                            progress.set_postfix({
                                'sym': symbol[:6],
                                'mod': model_name[:12],
                                'test': test_name[:14],
                                'run': run_id
                            })
                        progress.update(1)

        progress.close()
        self.reporter.summary()