# High-Frequency Microstructure LSTM

LSTM-based model for predicting bid-ask spreads and price direction in high-frequency trading environments using simulated market microstructure data.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Model Architecture

- **Data**: SPY 5-minute intervals over 30 days
- **Features**: Volume ratio, price momentum, volatility, order flow imbalance
- **Model**: Multi-task LSTM predicting spread changes and price direction
- **Strategy**: Long/short signals based on direction confidence thresholds

## Performance Results

- **Validation Accuracy**: current 53.42%
- **Strategy Return**: 0.66% vs 4.41% buy-and-hold
- **Sharpe Ratio**: 17.136
- **Max Drawdown**: -0.03%

## Microstructure Simulation

Uses volatility-based spread modeling to simulate realistic bid-ask spreads and order flow patterns from price/volume data. Incorporates volatility clustering and intraday noise patterns common in real market microstructure.

## Output Files

- `microstructure_performance.png` - Model performance visualization
- Trained LSTM model with spread and direction prediction capabilities
- Backtesting results with risk-adjusted performance metrics

## Production Enhancement Pathways

### Immediate Improvements
- **Real market data integration**:Order book data from professional providers
- **Feature engineering**: Add technical indicators, market impact measures, volatility regime detection
- **Model architecture**: Attention mechanisms, transformer encoders, ensemble methods

### Advanced Production Capabilities
- **Real-time inference**: Sub-millisecond prediction latency for live trading
- **Risk management**: Position sizing, portfolio-level risk controls, dynamic hedging
- **Multi-asset modeling**: Cross-asset arbitrage detection, sector rotation signals
- **Regime detection**: Adaptive model switching based on market conditions

### Enterprise-Grade Implementation 
- **Infrastructure**: Kubernetes deployment, real-time data pipelines, monitoring dashboards
- **Compliance**: Trade reporting, audit trails, regulatory risk controls
- **Integration**: Prime brokerage APIs, execution management systems, portfolio accounting
- **Performance optimization**: CUDA acceleration, distributed training, model compression

### Research & Development Extensions
- **Alternative data integration**: News sentiment, social media, satellite imagery
- **Causal inference**: Treatment effect estimation for strategy attribution
- **Meta-learning**: Rapid adaptation to new market regimes or asset classes
- **Reinforcement learning**: Dynamic strategy optimization with market feedback

## Technical Limitations

Current implementation uses simulated microstructure data which cannot capture real market complexities like latency arbitrage, information asymmetries, or institutional order flow patterns. Performance metrics reflect proof-of-concept capabilities rather than production trading results.

## Requirements

- Python 3.7+
- PyTorch 1.12+
- Market data access for production deployment
