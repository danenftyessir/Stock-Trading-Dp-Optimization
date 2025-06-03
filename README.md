# Stock Trading Dynamic Programming Optimization

A comprehensive Python framework for optimizing stock trading strategies using Dynamic Programming with k-transaction constraints, integrated with Alpha Vantage API data.

## Overview

This project implements a robust Dynamic Programming algorithm to find optimal buy/sell sequences that maximize profit under k-transaction constraints. The system provides comprehensive backtesting, performance analysis, and comparison with baseline strategies.

**Key Problem**: Maximize trading profits for a single stock over a historical period with a constraint of at most 'k' buy-sell transaction pairs.

## 🚀 Key Features

- **Dynamic Programming Core**: Advanced k-transaction optimization algorithm with transaction costs
- **Multiple Data Sources**: Alpha Vantage API integration with intelligent caching
- **Comprehensive Backtesting**: Walk-forward analysis, out-of-sample testing, Monte Carlo simulation
- **Baseline Comparisons**: Buy & Hold, Moving Average Crossover, Momentum strategies
- **Technical Indicators**: RSI, MACD, Bollinger Bands, SMA/EMA, and more
- **Risk Analysis**: Sharpe ratio, Sortino ratio, Maximum Drawdown, VaR, CVaR
- **Parameter Optimization**: Grid search and sensitivity analysis for k-values
- **ARIMA Forecasting**: Optional price prediction capabilities
- **CLI Automation**: Complete command-line workflow tools
- **Interactive Visualizations**: Performance charts, drawdown analysis, strategy comparison

## 📦 Installation

### Prerequisites
- Python 3.8+
- Alpha Vantage API key (free tier: 25 calls/day)

### Quick Setup
```bash
# Clone repository
git clone https://github.com/danendra/stock-trading-dp-optimization.git
cd stock-trading-dp-optimization

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Configure API key in config.yaml
api:
  alpha_vantage:
    api_key: "YOUR_API_KEY_HERE"
```

## 🔧 Quick Start

### Programmatic Usage
```python
from src import DynamicProgrammingTrader, AlphaVantageClient

# Initialize API client
client = AlphaVantageClient(api_key="YOUR_API_KEY")

# Download stock data
stock_df = client.get_stock_data("AAPL")
prices = stock_df['adjusted_close'].tolist()

# Create DP trader and optimize
trader = DynamicProgrammingTrader(max_transactions=5, transaction_cost=0.001)
max_profit_pct, trades = trader.optimize_profit(prices)

print(f"Maximum Profit: {max_profit_pct:.2%}")
print(f"Number of Trades: {len(trades)}")

# Run comprehensive backtest
backtest_results = trader.backtest(prices, dates=stock_df.index)
print(f"Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.3f}")
```

### CLI Usage
```bash
# Data collection
python scripts/data_collection.py --symbols AAPL GOOGL --config config.yaml

# Run backtesting
python scripts/run_backtesting.py --strategy dp --k-values 2 5 --symbols AAPL

# Parameter optimization
python scripts/run_backtesting.py --optimization --k-range 1 15 --objective sharpe_ratio

# Generate reports
python scripts/generate_reports.py --latest --format html markdown

# Full pipeline
python scripts/full_pipeline.py --config config.yaml
```

## 📁 Project Structure

```
stock-trading-dp-optimization/
├── src/                      # Core source code
│   ├── data/                # Data acquisition & preprocessing
│   ├── models/              # Trading strategies (DP, baseline, ARIMA)
│   ├── optimization/        # Backtesting & parameter optimization
│   ├── analysis/            # Performance & risk analysis
│   ├── visualization/       # Plotting utilities
│   └── utils/              # Logging & helper functions
├── scripts/                 # CLI automation tools
├── data/                    # Raw & processed data storage
├── results/                 # Backtesting & analysis outputs
├── reports/                 # Generated reports & visualizations
├── models/                  # Saved model parameters
├── tests/                   # Unit & integration tests
└── config.yaml             # Main configuration file
```

## 🧪 Core Components

### Dynamic Programming Algorithm
- **State Representation**: `buy[i][t]` and `sell[i][t]` for optimal profits
- **Transaction Costs**: Realistic trading cost integration
- **Trade Reconstruction**: Detailed buy/sell sequence generation
- **Unlimited Transactions**: Efficient greedy approach when k ≥ n/2

### Backtesting Framework
- **Simple Split**: Train/test validation
- **Walk-Forward**: Rolling window analysis
- **Out-of-Sample**: Three-way split with validation
- **Monte Carlo**: Bootstrap simulation for robustness

### Performance Metrics
- Return metrics: Total, annualized, risk-adjusted
- Risk metrics: Volatility, VaR, CVaR, Maximum Drawdown
- Trading metrics: Win rate, profit factor, Sharpe/Sortino ratios

## 📊 Example Results

The framework typically demonstrates:
- **DP Strategy**: Optimal historical performance with k-constraint adherence
- **vs Buy & Hold**: Often outperforms in volatile markets
- **vs MA Crossover**: Better risk-adjusted returns
- **Sensitivity Analysis**: Optimal k-values typically range 2-8 for daily data

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest -v tests/test_dpt_algorithm.py

# Test with coverage
pytest --cov=src tests/
```

## 📝 Configuration

Key parameters in `config.yaml`:
- **API Settings**: Alpha Vantage credentials and rate limits
- **Data Range**: Stock symbols, date ranges, cache settings
- **Strategy Parameters**: k-values, transaction costs, baseline configurations
- **Backtesting**: Capital allocation, train/test splits, risk-free rate

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Danendra Shafi Athallah**
- Email: danendra1967@gmail.com
- Institution: Institut Teknologi Bandung
- GitHub: [@danendra](https://github.com/danendra)

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{athallah2024dp,
  author = {Danendra Shafi Athallah},
  title = {Dynamic Programming Approach for Optimizing Stock Trading Profits with K-Transactions},
  institution = {Institut Teknologi Bandung},
  year = {2024},
  url = {https://github.com/danendra/stock-trading-dp-optimization}
}
```

## 🔗 Links

- [Documentation](docs/)
- [API Reference](https://docs.anthropic.com)
- [Issue Tracker](https://github.com/danendra/stock-trading-dp-optimization/issues)
- [Alpha Vantage API](https://www.alphavantage.co/support/#api-key)

---
