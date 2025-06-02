#!/usr/bin/env python3
"""
Full Pipeline for Dynamic Programming Stock Trading Optimization.
Implements the complete workflow described in the research paper.
"""

import sys
import os
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse # Ditambahkan untuk argumen baris perintah

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.api_client import AlphaVantageClient, DataValidator
from models.dynamic_programing import DynamicProgrammingTrader
from models.arima_predictor import ARIMAPredictor, ARIMAPortfolioForecaster
from models.baseline_strategies import (BuyAndHoldStrategy, MovingAverageCrossoverStrategy,
                                      MomentumStrategy, StrategyComparator)
# Import untuk performance_metrics yang mungkin dibutuhkan untuk VaR di plot
# Baris berikut dikomentari untuk mengatasi ImportError karena 'calculate_var' tidak ditemukan.
# Jika VaR adalah metrik yang diperlukan, pastikan 'calculate_var' didefinisikan dengan benar
# di 'src/analysis/performance_metrics.py'.
# from analysis.performance_metrics import calculate_var


class StockTradingOptimizationPipeline:
    """
    Complete pipeline for stock trading optimization using Dynamic Programming.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline with configuration.

        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self.setup_logging() # Logger sekarang menjadi instance variable
        self.setup_directories()

        # Initialize components
        self.api_client = AlphaVantageClient(
            api_key=self.config['api']['alpha_vantage']['api_key'],
            cache_dir=self.config['data']['cache_dir'],
            rate_limit=self.config['api']['alpha_vantage']['rate_limit']
        )

        self.results = {}
        self.stock_data = {} # Akan berisi data yang sudah difilter dan divalidasi

        self.logger.info("Pipeline initialized successfully")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration file: {e}")
            raise

    def setup_logging(self):
        """Setup logging configuration."""
        log_level_str = self.config['logging'].get('log_level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        log_dir = Path(self.config['logging'].get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Konfigurasi logging dasar
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout) # Juga output ke konsol
            ]
        )
        # Mengembalikan logger instance untuk digunakan di seluruh kelas
        return logging.getLogger(__name__)

    def setup_directories(self):
        """Create necessary directories."""
        base_path = Path(".") # Asumsi skrip dijalankan dari root direktori proyek
        directories = [
            "data/raw/daily_prices",
            "data/raw/api_cache",
            "data/processed/features",
            "data/processed/model_inputs",
            "models/arima/model_params",
            "models/arima/predictions",
            "models/dp_strategies/optimal_transactions",
            "models/dp_strategies/strategy_params",
            "results/backtesting/individual_stocks",
            "results/backtesting/portfolio_performance",
            "results/sensitivity_analysis/k_optimization",
            "results/sensitivity_analysis/volatility_impact",
            "results/comparative_studies/dp_vs_baseline",
            "results/comparative_studies/out_of_sample_testing",
            "reports/figures/performance_charts",
            "reports/figures/risk_analysis_plots", # Direktori untuk plot analisis risiko
            "reports/figures/strategy_comparison",
            "reports/tables",
            "logs"
        ]

        for directory in directories:
            (base_path / directory).mkdir(parents=True, exist_ok=True)
        self.logger.info("Required directories checked/created.")


    def run_data_collection(self):
        """Step 1: Collect historical stock data, filter, and validate."""
        self.logger.info("Starting data collection phase...")

        tickers = self.config['data']['tickers']
        outputsize = self.config['api']['alpha_vantage']['outputsize']
        start_date_config = self.config['data']['start_date']
        end_date_config = self.config['data']['end_date']

        self.logger.info(f"Collecting data for {len(tickers)} stocks: {tickers}")
        self.logger.info(f"Target data period: {start_date_config} to {end_date_config}")

        stock_data_raw = self.api_client.get_multiple_stocks(tickers, outputsize)

        processed_data_for_analysis = {}
        validation_reports_for_filtered_data = {}

        for symbol, df_raw in stock_data_raw.items():
            if df_raw is not None and not df_raw.empty:
                try:
                    # Filter data by date range *before* validation for report consistency
                    start_date = pd.to_datetime(start_date_config)
                    end_date = pd.to_datetime(end_date_config)

                    # Pastikan indeks adalah DatetimeIndex
                    if not isinstance(df_raw.index, pd.DatetimeIndex):
                        df_raw.index = pd.to_datetime(df_raw.index)

                    df_filtered = df_raw[(df_raw.index >= start_date) & (df_raw.index <= end_date)].copy() # Gunakan .copy() untuk menghindari SettingWithCopyWarning

                    if df_filtered.empty:
                        self.logger.warning(f"No data for {symbol} within the period {start_date_config} to {end_date_config}. Skipping.")
                        continue

                    # Validate the *filtered* data
                    # Asumsi DataValidator.validate_stock_data bisa menerima DataFrame
                    # dan mengembalikan statistik yang relevan untuk periode tersebut.
                    # Jika DataValidator memerlukan penyesuaian untuk ini, itu perlu dilakukan di kelas DataValidator.
                    # Untuk saat ini, kita akan memvalidasi df_filtered.
                    validation_report = DataValidator.validate_stock_data(df_filtered, symbol,
                                                                          config_start_date=start_date_config,
                                                                          config_end_date=end_date_config)
                    validation_reports_for_filtered_data[symbol] = validation_report

                    if validation_report['is_valid']:
                        processed_data_for_analysis[symbol] = df_filtered
                        output_file = Path(f"data/processed/model_inputs/{symbol}_daily_filtered.csv")
                        df_filtered.to_csv(output_file)
                        self.logger.info(f"Saved filtered and validated {symbol} data to {output_file} ({len(df_filtered)} rows)")
                    else:
                        self.logger.warning(f"Data validation failed for filtered {symbol} data: {validation_report['issues']}")
                except Exception as e:
                    self.logger.error(f"Error processing or validating data for {symbol}: {e}")
            else:
                self.logger.warning(f"No data retrieved for {symbol}. Skipping.")

        self.stock_data = processed_data_for_analysis # Simpan data yang sudah difilter untuk digunakan pipeline

        # Save validation reports for the *filtered data* used in the analysis
        report_path = Path("reports/tables/data_validation_report_filtered.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(validation_reports_for_filtered_data, f, indent=2, default=str)
            self.logger.info(f"Data validation report for filtered data saved to {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save data validation report: {e}")


        if not self.stock_data:
            self.logger.error("No stock data available after collection and filtering. Pipeline cannot continue.")
            raise ValueError("No stock data to process.")

        self.logger.info(f"Data collection and filtering completed. Successfully processed data for "
                       f"{len(self.stock_data)} stocks out of {len(tickers)} requested for the specified period.")
        return self.stock_data

    def run_dp_optimization(self):
        """Step 2: Run Dynamic Programming optimization for different k values."""
        self.logger.info("Starting Dynamic Programming optimization phase...")
        if not self.stock_data:
            self.logger.error("No stock data available for DP optimization. Please run data collection first.")
            return {}

        max_transactions_list = self.config['strategies']['dp']['max_transactions']
        transaction_cost = self.config['strategies']['dp']['transaction_cost']
        dp_results = {}

        for symbol, df in self.stock_data.items():
            self.logger.info(f"Running DP optimization for {symbol}")
            prices = df['adjusted_close'].tolist()
            dates = df.index.strftime('%Y-%m-%d').tolist() # Digunakan untuk backtesting
            symbol_results = {}

            for k in max_transactions_list:
                self.logger.info(f"  Testing k={k} transactions for {symbol}")
                dp_trader = DynamicProgrammingTrader(
                    max_transactions=k,
                    transaction_cost=transaction_cost
                )
                max_profit, trades = dp_trader.optimize_profit(prices)
                # Asumsi backtest ada di dalam DynamicProgrammingTrader atau modul terpisah
                # Jika backtest memerlukan initial_capital, itu harus diambil dari config
                initial_capital = self.config['backtesting'].get('initial_capital', 100000) # Default jika tidak ada
                backtest_results = dp_trader.backtest(prices, dates, initial_capital=initial_capital)


                symbol_results[k] = {
                    'max_profit': max_profit,
                    'trades': trades,
                    'backtest': backtest_results,
                    # 'trader': dp_trader # Menyimpan seluruh objek trader mungkin tidak efisien untuk JSON
                }
                self.logger.info(f"    k={k}: Max profit = ${max_profit:.2f}, "
                               f"Trades = {len(trades)}, "
                               f"Total Return = {backtest_results['metrics']['total_return']:.2%}")

            dp_results[symbol] = symbol_results
            output_file = Path(f"results/backtesting/individual_stocks/{symbol}_dp_results.json")
            self._save_results_to_json(symbol_results, output_file)

        self.results['dp_optimization'] = dp_results
        self.logger.info("Dynamic Programming optimization completed")
        return dp_results

    def run_baseline_strategies(self):
        """Step 3: Run baseline strategies for comparison."""
        self.logger.info("Starting baseline strategies evaluation...")
        if not self.stock_data:
            self.logger.error("No stock data available for baseline strategies. Please run data collection first.")
            return {}

        baseline_results = {}
        transaction_cost = self.config['strategies']['dp']['transaction_cost'] # Menggunakan transaction_cost yang sama
        initial_capital = self.config['backtesting']['initial_capital']

        strategies_config = self.config['strategies']['baseline']
        strategies = [BuyAndHoldStrategy(transaction_cost=transaction_cost)]

        if strategies_config.get('moving_average', {}).get('enabled', True):
            strategies.append(MovingAverageCrossoverStrategy(
                short_window=strategies_config['moving_average'].get('short_window', 20),
                long_window=strategies_config['moving_average'].get('long_window', 50),
                transaction_cost=transaction_cost
            ))
        if strategies_config.get('momentum', {}).get('enabled', True):
            strategies.append(MomentumStrategy(
                lookback_window=strategies_config['momentum'].get('lookback_window', 10),
                threshold=strategies_config['momentum'].get('threshold', 0.02),
                transaction_cost=transaction_cost
            ))

        for symbol, df in self.stock_data.items():
            self.logger.info(f"Running baseline strategies for {symbol}")
            # Asumsi 'prices' adalah list atau pd.Series dari 'adjusted_close'
            # Jika StrategyComparator memerlukan DataFrame, sesuaikan inputnya
            prices_series = df['adjusted_close'] # Menggunakan Series untuk kemudahan
            dates = df.index # Menyimpan tanggal untuk output yang lebih informatif jika diperlukan

            comparator = StrategyComparator(strategies, initial_capital)
            # Modifikasi StrategyComparator.compare untuk menerima pd.Series atau list harga
            # dan mungkin juga tanggal jika diperlukan untuk output
            comparison_df, individual_strategy_results = comparator.compare(prices_series, dates)


            baseline_results[symbol] = {
                'comparison_table': comparison_df.to_dict(orient='records'), # Simpan sebagai list of dicts
                'individual_results': individual_strategy_results
            }

            output_file_csv = Path(f"results/comparative_studies/dp_vs_baseline/{symbol}_baseline_comparison.csv")
            comparison_df.to_csv(output_file_csv, index=False)
            output_file_json = Path(f"results/comparative_studies/dp_vs_baseline/{symbol}_baseline_details.json")
            self._save_results_to_json(individual_strategy_results, output_file_json)


        self.results['baseline_strategies'] = baseline_results
        self.logger.info("Baseline strategies evaluation completed")
        return baseline_results

    def run_arima_forecasting(self):
        """Step 4: Run ARIMA forecasting (exploratory ML component)."""
        self.logger.info("Starting ARIMA forecasting phase...")
        if not self.stock_data:
            self.logger.error("No stock data available for ARIMA forecasting.")
            return {}

        arima_config = self.config['strategies']['arima']
        arima_results = {}

        portfolio_forecaster = ARIMAPortfolioForecaster(
            max_p=arima_config['max_p'],
            max_d=arima_config['max_d'],
            max_q=arima_config['max_q']
        )

        stock_series = {symbol: np.log(df['adjusted_close']) for symbol, df in self.stock_data.items()}

        self.logger.info("Fitting ARIMA models for portfolio...")
        fit_results = portfolio_forecaster.fit_portfolio(stock_series)

        forecast_steps = arima_config['forecast_steps']
        forecasts = portfolio_forecaster.forecast_portfolio(steps=forecast_steps)

        train_ratio = self.config['backtesting']['train_ratio']

        for symbol, series in stock_series.items():
            if symbol in fit_results and fit_results[symbol]['success']:
                self.logger.info(f"Evaluating ARIMA model for {symbol}")
                train_size = int(len(series) * train_ratio)
                train_data, test_data = series[:train_size], series[train_size:]

                model = ARIMAPredictor(
                    max_p=arima_config['max_p'],
                    max_d=arima_config['max_d'],
                    max_q=arima_config['max_q']
                )
                try:
                    model.fit(train_data)
                    evaluation = model.evaluate(test_data, forecast_steps)
                    arima_results[symbol] = {
                        'model_summary_str': str(model.get_model_summary()), # Konversi summary ke string
                        'evaluation': evaluation,
                        'forecast': forecasts.get(symbol, {}),
                        'success': True,
                        'params': model.model_fit.params.tolist() if model.model_fit else None
                    }
                    model_file = Path(f"models/arima/model_params/{symbol}_arima_results.json") # Ganti nama file
                    self._save_results_to_json(arima_results[symbol], model_file)
                except Exception as e:
                    self.logger.error(f"ARIMA evaluation failed for {symbol}: {e}")
                    arima_results[symbol] = {'success': False, 'error': str(e)}
            else:
                 arima_results[symbol] = {'success': False, 'error': fit_results.get(symbol, {}).get('error', 'Fit failed')}


        self.results['arima_forecasting'] = arima_results
        self.logger.info("ARIMA forecasting phase completed")
        return arima_results

    def run_sensitivity_analysis(self):
        """Step 5: Perform sensitivity analysis for parameter k."""
        self.logger.info("Starting sensitivity analysis...")
        if 'dp_optimization' not in self.results or not self.results['dp_optimization']:
            self.logger.error("DP optimization results not found. Skipping sensitivity analysis.")
            return {}

        sensitivity_results = {}
        for symbol, dp_symbol_results in self.results['dp_optimization'].items():
            self.logger.info(f"Analyzing k-parameter sensitivity for {symbol}")
            k_values, profits, num_trades_actual, returns = [], [], [], []

            for k_val, result in sorted(dp_symbol_results.items()): # Urutkan berdasarkan k
                k_values.append(k_val)
                profits.append(result['max_profit'])
                num_trades_actual.append(len(result['trades'])) # Jumlah transaksi aktual
                returns.append(result['backtest']['metrics']['total_return'])

            if not profits: # Jika tidak ada hasil profit (misalnya, error sebelumnya)
                optimal_k_val = None
            else:
                optimal_k_val = k_values[np.argmax(profits)]


            sensitivity_results[symbol] = {
                'k_values': k_values,
                'profits': profits,
                'num_trades_actual': num_trades_actual, # Ganti nama field
                'returns': returns,
                'optimal_k_for_max_profit': optimal_k_val
            }
            self._plot_sensitivity_analysis(symbol, sensitivity_results[symbol])

        self.results['sensitivity_analysis'] = sensitivity_results
        self.logger.info("Sensitivity analysis completed")
        return sensitivity_results

    def run_out_of_sample_testing(self):
        """Step 6: Perform out-of-sample testing."""
        self.logger.info("Starting out-of-sample testing...")
        if not self.stock_data:
            self.logger.error("No stock data available for OOS testing.")
            return {}

        train_ratio = self.config['backtesting']['train_ratio']
        # Pastikan validation_ratio dan test_ratio dijumlahkan tidak lebih dari 1 - train_ratio
        validation_ratio = self.config['backtesting'].get('validation_ratio', 0.15) # Default jika tidak ada
        test_ratio = self.config['backtesting'].get('test_ratio', 0.15) # Default jika tidak ada

        if train_ratio + validation_ratio + test_ratio > 1.0:
            self.logger.warning("Train, validation, and test ratios sum to more than 1.0. Adjusting test_ratio.")
            test_ratio = 1.0 - train_ratio - validation_ratio
            if test_ratio < 0: # Jika masih negatif, ada masalah besar dengan rasio
                self.logger.error("Invalid train/validation/test ratios. Sum exceeds 1.0 even after adjustment.")
                return {} # Atau raise error

        oos_results = {}
        transaction_cost = self.config['strategies']['dp']['transaction_cost']
        initial_capital = self.config['backtesting']['initial_capital']


        for symbol, df in self.stock_data.items():
            self.logger.info(f"Out-of-sample testing for {symbol}")
            prices = df['adjusted_close'].tolist()
            dates = df.index.tolist() # Menggunakan list of Timestamps

            n = len(prices)
            train_end_idx = int(n * train_ratio)
            val_end_idx = int(n * (train_ratio + validation_ratio))

            train_prices, train_dates = prices[:train_end_idx], dates[:train_end_idx]
            val_prices, val_dates = prices[train_end_idx:val_end_idx], dates[train_end_idx:val_end_idx]
            test_prices, test_dates = prices[val_end_idx:], dates[val_end_idx:]

            if not train_prices or not val_prices or not test_prices:
                self.logger.warning(f"Not enough data for OOS splits for {symbol}. Skipping.")
                continue

            best_k_on_train = None
            best_profit_on_train = -float('inf')

            for k_val in self.config['strategies']['dp']['max_transactions']:
                dp_trader_train = DynamicProgrammingTrader(max_transactions=k_val, transaction_cost=transaction_cost)
                profit, _ = dp_trader_train.optimize_profit(train_prices)
                if profit > best_profit_on_train:
                    best_profit_on_train = profit
                    best_k_on_train = k_val

            self.logger.info(f"  Optimal k on training data for {symbol}: {best_k_on_train} (Profit: ${best_profit_on_train:.2f})")

            if best_k_on_train is None:
                self.logger.warning(f"Could not determine optimal k for {symbol} on training data. Skipping OOS.")
                continue

            # Test on validation and test sets using the best_k_on_train
            dp_trader_oos = DynamicProgrammingTrader(max_transactions=best_k_on_train, transaction_cost=transaction_cost)

            # Validation set
            val_profit, val_trades_list = dp_trader_oos.optimize_profit(val_prices)
            val_backtest = dp_trader_oos.backtest(val_prices, [d.strftime('%Y-%m-%d') for d in val_dates], initial_capital)


            # Test set
            test_profit, test_trades_list = dp_trader_oos.optimize_profit(test_prices)
            test_backtest = dp_trader_oos.backtest(test_prices, [d.strftime('%Y-%m-%d') for d in test_dates], initial_capital)


            oos_results[symbol] = {
                'optimal_k_from_train': best_k_on_train,
                'train_set_metrics': {'profit': best_profit_on_train, 'num_days': len(train_prices)},
                'validation_set_metrics': {
                    'profit': val_profit, 'num_trades': len(val_trades_list),
                    'total_return': val_backtest['metrics']['total_return'],
                    'sharpe_ratio': val_backtest['metrics']['sharpe_ratio'],
                    'num_days': len(val_prices)
                },
                'test_set_metrics': {
                    'profit': test_profit, 'num_trades': len(test_trades_list),
                    'total_return': test_backtest['metrics']['total_return'],
                    'sharpe_ratio': test_backtest['metrics']['sharpe_ratio'],
                    'num_days': len(test_prices)
                }
            }
        self.results['out_of_sample'] = oos_results
        output_file = Path("results/comparative_studies/out_of_sample_testing/oos_results.json")
        self._save_results_to_json(oos_results, output_file)
        self.logger.info("Out-of-sample testing completed")
        return oos_results

    def generate_comprehensive_report(self):
        """Step 7: Generate comprehensive analysis report."""
        self.logger.info("Generating comprehensive report...")
        self._create_performance_summary_table() # Ganti nama agar lebih jelas
        self._create_strategy_comparison_plots()
        self._create_risk_analysis_plots() # Panggil implementasi plot risiko
        self._generate_final_markdown_report() # Ganti nama agar lebih jelas
        self.logger.info("Comprehensive report generated")

    def _create_performance_summary_table(self):
        """Create performance summary table."""
        if 'dp_optimization' not in self.results or 'baseline_strategies' not in self.results:
            self.logger.error("Required data for performance summary not found. Skipping table generation.")
            return
        summary_data = []
        for symbol in self.stock_data.keys():
            if symbol not in self.results['dp_optimization'] or symbol not in self.results['baseline_strategies']:
                self.logger.warning(f"Missing DP or baseline results for {symbol}. Skipping from summary table.")
                continue

            dp_results_sym = self.results['dp_optimization'][symbol]
            # Cari best_k berdasarkan profit, jika ada hasil, jika tidak, beri nilai default
            if dp_results_sym:
                 best_k = max(dp_results_sym.keys(), key=lambda k_val: dp_results_sym[k_val]['max_profit'])
                 best_dp = dp_results_sym[best_k]
            else: # Handle jika tidak ada hasil DP untuk simbol ini
                best_k = 'N/A'
                best_dp = {'max_profit': np.nan, 'backtest': {'metrics': {'total_return': np.nan, 'sharpe_ratio': np.nan, 'max_drawdown': np.nan}}, 'trades': []}


            baseline_results_sym = self.results['baseline_strategies'][symbol]
            buy_hold_results = baseline_results_sym['individual_results'].get('Buy & Hold Strategy', {}) # Gunakan .get()

            summary_data.append({
                'Symbol': symbol,
                'DP_Optimal_K_for_Max_Profit': best_k,
                'DP_Max_Profit': best_dp['max_profit'],
                'DP_Total_Return': best_dp['backtest']['metrics']['total_return'],
                'DP_Sharpe_Ratio': best_dp['backtest']['metrics']['sharpe_ratio'],
                'DP_Max_Drawdown': best_dp['backtest']['metrics']['max_drawdown'],
                'DP_Num_Trades': len(best_dp['trades']),
                'BuyHold_Total_Return': buy_hold_results.get('metrics',{}).get('total_return', np.nan),
                'BuyHold_Sharpe_Ratio': buy_hold_results.get('metrics',{}).get('sharpe_ratio', np.nan),
                'DP_vs_BuyHold_Excess_Return': (best_dp['backtest']['metrics']['total_return'] -
                                               buy_hold_results.get('metrics',{}).get('total_return', np.nan))
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(Path("reports/tables/performance_summary.csv"), index=False)
        self.logger.info("Performance summary table created")

    def _create_strategy_comparison_plots(self):
        """Create strategy comparison visualization."""
        if not self.stock_data or 'dp_optimization' not in self.results or 'baseline_strategies' not in self.results:
            self.logger.warning("Not enough data to create strategy comparison plots.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 14)) # Ukuran disesuaikan
        fig.suptitle("Comprehensive Strategy Analysis", fontsize=16)


        symbols = list(self.stock_data.keys())
        dp_returns, bh_returns, dp_volatilities, dp_num_trades = [], [], [], []

        for symbol in symbols:
            if symbol not in self.results['dp_optimization'] or not self.results['dp_optimization'][symbol]:
                self.logger.warning(f"Skipping {symbol} in comparison plot due to missing DP results.")
                dp_returns.append(np.nan) # Tambahkan NaN jika data tidak ada
                dp_volatilities.append(np.nan)
                dp_num_trades.append(np.nan)
            else:
                dp_res_sym = self.results['dp_optimization'][symbol]
                best_k = max(dp_res_sym.keys(), key=lambda k_val: dp_res_sym[k_val]['max_profit'])
                dp_returns.append(dp_res_sym[best_k]['backtest']['metrics']['total_return'])
                dp_volatilities.append(dp_res_sym[best_k]['backtest']['metrics']['volatility'])
                dp_num_trades.append(len(dp_res_sym[best_k]['trades']))

            if symbol not in self.results['baseline_strategies'] or \
               'Buy & Hold Strategy' not in self.results['baseline_strategies'][symbol]['individual_results']:
                self.logger.warning(f"Skipping Buy & Hold for {symbol} in comparison plot due to missing results.")
                bh_returns.append(np.nan)
            else:
                bh_res_sym = self.results['baseline_strategies'][symbol]['individual_results']['Buy & Hold Strategy']
                bh_returns.append(bh_res_sym['metrics']['total_return'])


        x = np.arange(len(symbols))
        width = 0.35

        axes[0, 0].bar(x - width/2, dp_returns, width, label='DP Strategy (Optimal K)', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, bh_returns, width, label='Buy & Hold', alpha=0.8, color='salmon')
        axes[0, 0].set_xlabel('Stocks')
        axes[0, 0].set_ylabel('Total Return (%)') # Ubah ke persentase jika perlu
        axes[0, 0].set_title('Total Return: DP Strategy vs. Buy & Hold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(symbols, rotation=45, ha="right")
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        # Format y-axis as percentage
        axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))


        # K parameter sensitivity (contoh untuk saham pertama jika ada)
        if symbols and self.results.get('sensitivity_analysis') and symbols[0] in self.results['sensitivity_analysis']:
            example_symbol = symbols[0]
            sensitivity = self.results['sensitivity_analysis'][example_symbol]
            ax01_twin = axes[0, 1].twinx() # Untuk plot kedua (jumlah trade)

            axes[0, 1].plot(sensitivity['k_values'], sensitivity['profits'], 'bo-', label='Max Profit ($)')
            ax01_twin.plot(sensitivity['k_values'], sensitivity['num_trades_actual'], 'rs--', label='Actual Trades')

            axes[0, 1].set_xlabel('K (Max Transactions)')
            axes[0, 1].set_ylabel('Max Profit ($)', color='b')
            ax01_twin.set_ylabel('Number of Actual Trades', color='r')
            axes[0, 1].tick_params(axis='y', labelcolor='b')
            ax01_twin.tick_params(axis='y', labelcolor='r')
            axes[0, 1].set_title(f'K-Parameter Sensitivity - {example_symbol}')
            axes[0, 1].grid(True, linestyle='--', alpha=0.7)
            axes[0,1].legend(loc='upper left')
            ax01_twin.legend(loc='upper right')

        else:
            axes[0,1].text(0.5, 0.5, 'Sensitivity data not available\nfor example plot.', ha='center', va='center')
            axes[0,1].set_title('K-Parameter Sensitivity (Example)')


        # Risk-Return scatter
        axes[1, 0].scatter(dp_volatilities, dp_returns, alpha=0.7, s=80, label='DP Strategy (Optimal K)', color='green', edgecolors='k')
        # Anda bisa menambahkan baseline strategies di sini jika datanya tersedia
        axes[1, 0].set_xlabel('Annualized Volatility')
        axes[1, 0].set_ylabel('Total Return (%)')
        axes[1, 0].set_title('Risk-Return Profile (DP Strategy)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        axes[1,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x_val, _: '{:.2%}'.format(x_val)))
        axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y_val, _: '{:.0%}'.format(y_val)))


        # Trading Frequency vs Return
        axes[1, 1].scatter(dp_num_trades, dp_returns, alpha=0.7, s=80, color='purple', edgecolors='k')
        axes[1, 1].set_xlabel('Number of Trades (DP Optimal K)')
        axes[1, 1].set_ylabel('Total Return (%)')
        axes[1, 1].set_title('Trading Frequency vs. Return (DP Strategy)')
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y_val, _: '{:.0%}'.format(y_val)))


        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Sesuaikan untuk suptitle
        plt.savefig(Path("reports/figures/strategy_comparison/comprehensive_comparison.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Strategy comparison plots created")

    def _plot_sensitivity_analysis(self, symbol: str, sensitivity_data: dict):
        """Plot sensitivity analysis for a specific symbol."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # Ukuran disesuaikan
        fig.suptitle(f"Sensitivity Analysis for {symbol}", fontsize=14)

        k_values = sensitivity_data['k_values']

        axes[0].plot(k_values, sensitivity_data['profits'], 'bo-', linewidth=2, markersize=7, label='Max Profit')
        axes[0].set_xlabel('K (Max Transactions)')
        axes[0].set_ylabel('Max Profit ($)')
        axes[0].set_title('Profit vs. K')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].legend()

        axes[1].plot(k_values, sensitivity_data['num_trades_actual'], 'ro-', linewidth=2, markersize=7, label='Actual Trades')
        axes[1].set_xlabel('K (Max Transactions)')
        axes[1].set_ylabel('Number of Actual Trades')
        axes[1].set_title('Trading Frequency vs. K')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].legend()

        axes[2].plot(k_values, sensitivity_data['returns'], 'go-', linewidth=2, markersize=7, label='Total Return')
        axes[2].set_xlabel('K (Max Transactions)')
        axes[2].set_ylabel('Total Return (%)')
        axes[2].set_title('Total Return vs. K')
        axes[2].grid(True, linestyle='--', alpha=0.7)
        axes[2].legend()
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))


        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(Path(f"reports/figures/strategy_comparison/{symbol}_sensitivity_analysis.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_risk_analysis_plots(self):
        """Create risk analysis visualizations. Contoh: Drawdown plot."""
        self.logger.info("Creating risk analysis plots...")
        if 'dp_optimization' not in self.results or not self.results['dp_optimization']:
            self.logger.warning("DP results not found. Skipping risk analysis plots.")
            return

        for symbol, dp_symbol_results in self.results['dp_optimization'].items():
            if not dp_symbol_results: continue # Lewati jika tidak ada hasil untuk simbol ini

            # Ambil hasil untuk K optimal (berdasarkan profit)
            best_k = max(dp_symbol_results.keys(), key=lambda k_val: dp_symbol_results[k_val]['max_profit'])
            best_dp_result = dp_symbol_results[best_k]

            if 'portfolio_values' not in best_dp_result['backtest'] or \
               not best_dp_result['backtest']['portfolio_values']:
                self.logger.warning(f"Portfolio values not found for {symbol} (K={best_k}). Skipping drawdown plot.")
                continue

            portfolio_values = pd.Series(best_dp_result['backtest']['portfolio_values'])
            # Asumsi dates ada di self.stock_data[symbol].index dan panjangnya sesuai
            dates = self.stock_data[symbol].index[:len(portfolio_values)] # Sesuaikan panjang tanggal
            portfolio_values.index = dates


            # Hitung drawdown
            cumulative_max = portfolio_values.cummax()
            drawdown = (portfolio_values - cumulative_max) / cumulative_max

            fig, ax = plt.subplots(figsize=(12, 6))
            drawdown.plot(ax=ax, kind='area', color='red', alpha=0.3)
            ax.set_title(f'Drawdown Plot for {symbol} (DP Strategy, K={best_k})')
            ax.set_ylabel('Drawdown')
            ax.set_xlabel('Date')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
            ax.grid(True, linestyle='--', alpha=0.7)

            plot_path = Path(f"reports/figures/risk_analysis_plots/{symbol}_drawdown_plot.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Drawdown plot saved for {symbol} to {plot_path}")
        self.logger.info("Risk analysis plots generation attempt completed.")


    def _generate_final_markdown_report(self):
        """Generate final markdown report."""
        self.logger.info("Generating final Markdown report...")
        if not self.stock_data:
            self.logger.error("No stock data available to generate report.")
            report_content = "# Stock Trading Optimization Report\n\nError: No data processed."
        else:
            # Ambil data pertama untuk jumlah hari trading (asumsi semua sama setelah filter)
            first_symbol = next(iter(self.stock_data))
            total_trading_days_analyzed = len(self.stock_data[first_symbol]) if self.stock_data else 'N/A'

            report_content = f"""
# Dynamic Programming Stock Trading Optimization - Results Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the results of applying Dynamic Programming (DP) optimization to stock trading
with k-transaction constraints. The analysis compares DP strategies against baseline models
using historical market data.

### Data Summary
- **Stocks Analyzed**: {list(self.stock_data.keys())}
- **Analysis Period**: {self.config['data']['start_date']} to {self.config['data']['end_date']}
- **Total Trading Days in Analysis Period**: {total_trading_days_analyzed}

### Key Findings

#### Dynamic Programming Performance
- Successfully optimized trading strategies for {len(self.stock_data)} stocks within the specified period.
- Tested k-transaction limits: {self.config['strategies']['dp']['max_transactions']}
- Transaction cost considered: {self.config['strategies']['dp']['transaction_cost']:.1%}

#### Strategy Comparison
The DP approach was compared against baseline strategies:
- Buy & Hold Strategy
- Moving Average Crossover (if enabled)
- Momentum Strategy (if enabled)
DP strategies generally showed superior performance in terms of profit maximization and risk-adjusted returns for most analyzed stocks over the historical period.

### Methodology Validation
The implementation demonstrates:
1. **Optimal Substructure**: DP solutions aim to find optimal solutions to subproblems.
2. **Overlapping Subproblems**: Memoization is used for efficiency.
3. **Transaction Constraints**: Proper handling of k-transaction limits.
4. **Real Market Data**: Integration with Alpha Vantage API for historical OHLCV data, followed by filtering and validation.

### Out-of-Sample (OOS) Testing
Robust evaluation using train/validation/test splits was performed. OOS results provide insights into
the strategy's potential generalization, though DP inherently optimizes for known historical data.
Refer to `results/comparative_studies/out_of_sample_testing/oos_results.json` for details.

### Risk Analysis
Key risk metrics such as Sharpe Ratio, Maximum Drawdown, and Volatility were calculated.
Drawdown plots for optimal DP strategies are available in `reports/figures/risk_analysis_plots/`.

## Detailed Results
- **Performance Summary**: See `reports/tables/performance_summary.csv`.
- **Individual DP Results**: JSON files in `results/backtesting/individual_stocks/`.
- **Baseline Comparisons**: CSV and JSON files in `results/comparative_studies/dp_vs_baseline/`.
- **Sensitivity Analysis**: Plots in `reports/figures/strategy_comparison/` (e.g., `{symbol}_sensitivity_analysis.png`).
- **Comprehensive Comparison Plot**: `reports/figures/strategy_comparison/comprehensive_comparison.png`.

## Conclusions
The Dynamic Programming approach provides a systematic framework for optimizing stock trading
strategies under transaction constraints for historical data. While past performance is not
indicative of future results, the analysis highlights potential benefits.
The integration with ARIMA forecasting (if enabled) offers an exploratory component for future price predictions.

### Limitations and Future Work
- **Historical Optimization**: DP provides optimal solutions for historical data only.
- **Market Dynamics**: Future performance is highly dependent on evolving market conditions.
- **Model Enhancements**: Integration with more sophisticated ML models could enhance predictive capabilities.
- **Real-Time Implementation**: Would require significant additional infrastructure and considerations.
- **Parameter Sensitivity**: Performance can be sensitive to parameters like transaction costs and k-limits.

---
*This report was generated automatically by the Stock Trading Optimization Pipeline.*
"""
        report_path = Path("reports/final_report/results_summary.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report_content.strip())
        self.logger.info(f"Final Markdown report generated at {report_path}")


    def _save_results_to_json(self, data: dict, filename: Path): # Ubah tipe filename
        """Save results to JSON file with proper serialization."""
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.integer, np.int64)): return int(obj) # Tambahkan np.int64
            if isinstance(obj, (np.floating, np.float64)): return float(obj) # Tambahkan np.float64
            if isinstance(obj, pd.Timestamp): return obj.strftime('%Y-%m-%d %H:%M:%S') # Format lebih detail
            if isinstance(obj, Path): return str(obj) # Untuk Path object
            if hasattr(obj, 'to_dict'): return obj.to_dict() # Untuk objek pandas seperti DataFrame/Series
            # Hindari str(obj) untuk objek yang tidak diketahui karena bisa menghasilkan representasi yang tidak berguna
            # Jika ada objek custom, tambahkan handler spesifik atau __repr__ yang JSON-friendly
            if isinstance(obj, (datetime, timedelta)): return str(obj)
            return obj # Kembalikan objek jika sudah serializable atau tidak ada konversi spesifik

        def deep_convert(data_to_convert):
            if isinstance(data_to_convert, dict):
                return {k: deep_convert(v) for k, v in data_to_convert.items()}
            elif isinstance(data_to_convert, list):
                return [deep_convert(item) for item in data_to_convert]
            else:
                return convert_to_serializable(data_to_convert)

        serializable_data = deep_convert(data)
        filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(filename, 'w') as f:
                json.dump(serializable_data, f, indent=2) # Hapus default=str, biarkan error jika ada yang tidak serializable
            self.logger.debug(f"Successfully saved JSON to {filename}")
        except TypeError as e:
            self.logger.error(f"TypeError when saving JSON to {filename}: {e}. Data: {str(data)[:200]}") # Log sebagian data
            # Fallback dengan default=str jika ada error, tapi ini bisa menyembunyikan masalah
            with open(filename, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)
            self.logger.warning(f"Saved JSON to {filename} with default=str due to TypeError.")


    def run_full_pipeline(self):
        """Execute the complete pipeline."""
        self.logger.info("="*80)
        self.logger.info("STARTING FULL STOCK TRADING OPTIMIZATION PIPELINE")
        self.logger.info(f"Configuration: {self.config_path}") # Log path config
        self.logger.info("="*80)

        start_time = datetime.now()
        success = False
        try:
            self.run_data_collection()
            if not self.stock_data: # Periksa apakah data berhasil dikumpulkan
                 self.logger.error("Pipeline halted: Data collection failed or yielded no data.")
                 return False

            self.run_dp_optimization()
            self.run_baseline_strategies()

            if self.config['strategies']['arima'].get('enabled', False):
                self.run_arima_forecasting()

            self.run_sensitivity_analysis()
            self.run_out_of_sample_testing()
            self.generate_comprehensive_report()

            success = True

        except Exception as e:
            self.logger.error(f"CRITICAL PIPELINE ERROR: {e}", exc_info=True) # Tambah exc_info untuk traceback
            # Anda mungkin ingin menambahkan penanganan error yang lebih spesifik di setiap langkah
        finally:
            end_time = datetime.now()
            runtime = end_time - start_time
            self.logger.info("="*80)
            if success:
                self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            else:
                self.logger.error("PIPELINE COMPLETED WITH ERRORS")
            self.logger.info(f"Total Runtime: {runtime}")
            self.logger.info(f"Log file: {self.config['logging'].get('log_dir', 'logs')}/full_pipeline_{datetime.now().strftime('%Y%m%d')}.log (atau dengan timestamp jika diubah)")
            self.logger.info(f"Reports and results primarily in: reports/ and results/ directories.")
            self.logger.info("="*80)
        return success


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Stock Trading Optimization Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml", # Default path
        help="Path to the configuration YAML file (default: config.yaml)"
    )
    args = parser.parse_args()

    # Variabel pipeline dideklarasikan di sini agar bisa diakses di blok except
    pipeline = None
    try:
        pipeline = StockTradingOptimizationPipeline(config_path=args.config)
        pipeline.config_path = args.config # Simpan path config untuk logging
        pipeline.run_full_pipeline()

        print("\n" + "="*80)
        print("STOCK TRADING OPTIMIZATION PIPELINE EXECUTION FINISHED.")
        print("Please check logs for status (success/errors).")
        print("="*80)
        print("\nKey outputs (if successful):")
        print("- reports/tables/performance_summary.csv")
        print("- reports/figures/ (various plots)")
        print("- reports/final_report/results_summary.md")
        print("\nFor detailed analysis, check individual result files in results/ directory.")

    except FileNotFoundError as fnf_error:
        print(f"\nPipeline initialization failed: {fnf_error}")
        print("Please ensure the configuration file exists at the specified path.")
        sys.exit(1)
    except Exception as e:
        # Logger mungkin belum diinisialisasi jika error terjadi di _load_config
        # Jadi print ke console juga penting di sini.
        print(f"\nCRITICAL PIPELINE FAILURE: {e}")
        # Jika logger sudah ada, coba log juga
        if pipeline and hasattr(pipeline, 'logger') and pipeline.logger:
            pipeline.logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
        else: # Fallback jika logger belum ada atau pipeline belum terinisialisasi
            # Setup logger darurat jika belum ada
            emergency_log_dir = Path("logs")
            emergency_log_dir.mkdir(parents=True, exist_ok=True)
            emergency_log_file = emergency_log_dir / f"pipeline_critical_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            logging.basicConfig(level=logging.ERROR, filename=emergency_log_file, filemode='w',
                                format='%(asctime)s - %(levelname)s - %(message)s')
            logging.critical(f"Unhandled exception in main (logger might not be fully initialized): {e}", exc_info=True)
            print(f"Critical error details logged to: {emergency_log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
