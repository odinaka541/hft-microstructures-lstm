# 541

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# set random seeds for reproducibility
np.random.seed(23)
torch.manual_seed(23)


def simulate_microstructure_data(price_data):
    """
    simulates bid-ask spreads and order flow imbalances from price/volume data
    uses volatility-based spread model common in academic literature
    """
    sim_df = price_data.copy()

    # calculating returns and volatility for spread simulation
    sim_df['returns'] = sim_df['close'].pct_change()
    sim_df['volatility'] = sim_df['returns'].rolling(window=20, min_periods=1).std()

    # simulating bid-ask spread based on volatility | higher volatility = wider spreads
    base_spread = 0.001  # 10 basis points base
    sim_df['bid_ask_spread'] = base_spread + (sim_df['volatility'] * 2.0)

    # simulating order flow imbalance | positive when buying pressure, negative when selling pressure
    sim_df['price_change'] = sim_df['close'].diff()
    sim_df['volume_weighted_price'] = (sim_df['high'] + sim_df['low'] + sim_df['close']) / 3
    sim_df['order_flow_imbalance'] = np.where(sim_df['price_change'] > 0,
                                          sim_df['volume'] * 0.6,  # more buying
                                          sim_df['volume'] * -0.4)  # more selling

    # adding realistic noise to spreads
    noise = np.random.normal(0, sim_df['bid_ask_spread'].std() * 0.1, len(sim_df))
    sim_df['bid_ask_spread'] += noise
    sim_df['bid_ask_spread'] = np.maximum(sim_df['bid_ask_spread'], 0.0001)  # minimum spread

    # creating features for lstm model
    sim_df['volume_ratio'] = sim_df['volume'] / sim_df['volume'].rolling(window=20, min_periods=1).mean()
    sim_df['price_momentum'] = sim_df['close'].pct_change(periods=5)
    sim_df['spread_ratio'] = sim_df['bid_ask_spread'] / sim_df['bid_ask_spread'].rolling(window=10, min_periods=1).mean()

    # dropping nas +  return clean data
    sim_df = sim_df.dropna()
    print(f"simulated microstructure data: {len(sim_df)} periods")

    return sim_df


def get_spy_data():
    """ fetch spy intraday data for last 30 days at 5min intervals"""
    try:
        # SPY data with 5-minute intervals for past 30 days
        ticker = yf.Ticker("SPY")
        data = ticker.history(period="1mo", interval="5m")

        # cleaning column names and prepare dataframe
        data.columns = data.columns.str.lower()
        data = data.reset_index()
        data.columns = data.columns.str.lower()

        print(f"fetched spy data: {len(data)} periods")
        return data

    except Exception as e:
        print(f"error fetching data: {e}")
        return None


class MicrostructureDataset(Dataset):
    """ dataset class for microstructure lstm training"""

    def __init__(self, features, targets, sequence_length=20):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        return (
            self.features[idx:idx + self.sequence_length],
            self.targets[idx + self.sequence_length]
        )


class MicrostructureLSTM(nn.Module):
    """ lstm model for microstructure prediction"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(MicrostructureLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # lstm layers with dropout for regularization
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # output layers for spread prediction and direction classification
        self.spread_head = nn.Linear(hidden_size, 1)
        self.direction_head = nn.Linear(hidden_size, 2)  # up/down classification
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # lstm forward pass
        lstm_out, _ = self.lstm(x)

        #last output for prediction
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)

        # predicting spread and price direction
        spread_pred = self.spread_head(last_output)
        direction_pred = self.direction_head(last_output)

        return spread_pred, direction_pred


def prepare_training_data(sim_df, sequence_length=20):
    """prepare features and targets for lstm training"""

    # select feature columns for model input
    feature_cols = ['volume_ratio', 'price_momentum', 'volatility', 'order_flow_imbalance']
    target_col = 'bid_ask_spread'

    # using min-max scaling to normalize features
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    features_normalized = scaler_features.fit_transform(sim_df[feature_cols])
    target_normalized = scaler_target.fit_transform(sim_df[[target_col]])

    # create directional labels (1 if spread increases, 0 if decreases)
    direction_labels = (sim_df[target_col].diff().shift(-1) > 0).astype(int)[:-1]

    return (features_normalized[:-1], target_normalized[:-1].flatten(),
            direction_labels.values, scaler_features, scaler_target)


def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    """train the lstm model with both spread and direction prediction"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # separating loss functions for regression and classification
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            spread_targets = batch_targets[:, 0].to(device)  # spread values
            direction_targets = batch_targets[:, 1].long().to(device)  # direction labels

            optimizer.zero_grad()

            # forward pass
            spread_pred, direction_pred = model(batch_features)

            # calculate combined loss
            spread_loss = mse_criterion(spread_pred.squeeze(), spread_targets)
            direction_loss = ce_criterion(direction_pred, direction_targets)
            total_loss_batch = spread_loss + direction_loss

            # backward pass and optimization
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()

        # validation accuracy calculation
        model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                direction_targets = batch_targets[:, 1].long().to(device)

                _, direction_pred = model(batch_features)
                _, predicted = torch.max(direction_pred.data, 1)

                total_predictions += direction_targets.size(0)
                correct_predictions += (predicted == direction_targets).sum().item()

        avg_loss = total_loss / len(train_loader)
        val_accuracy = correct_predictions / total_predictions

        train_losses.append(avg_loss)
        val_accuracies.append(val_accuracy)

        if epoch % 5 == 0:
            print(f'epoch [{epoch + 1}/{num_epochs}], loss: {avg_loss:.4f}, val_accuracy: {val_accuracy:.4f}')

    return train_losses, val_accuracies


def backtest_strategy(sim_df, model, scaler_features, scaler_target, sequence_length=20):
    """ basic bcktest trading strategy using lstm predictions"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # preparing test data
    feature_cols = ['volume_ratio', 'price_momentum', 'volatility', 'order_flow_imbalance']
    test_features = scaler_features.transform(sim_df[feature_cols])

    signals = []
    prices = []

    # generating trading signals using rolling predictions
    with torch.no_grad():
        for i in range(sequence_length, len(test_features) - 1):
            # get sequence for prediction
            seq = test_features[i - sequence_length:i]
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)

            # predict spread and direction
            spread_pred, direction_pred = model(seq_tensor)
            direction_prob = torch.softmax(direction_pred, dim=1)

            # trading logic: go long when direction confidence > 0.6, short when < 0.4
            direction_confidence = direction_prob[0, 1].item()  # probability of upward movement

            if direction_confidence > 0.6:
                signals.append(1)  # long signal
            elif direction_confidence < 0.4:
                signals.append(-1)  # short signal
            else:
                signals.append(0)  # no position

            prices.append(sim_df.iloc[i]['close'])

    # calculate strategy returns
    signals = np.array(signals)
    prices = np.array(prices)
    returns = np.diff(prices) / prices[:-1]

    # strategy returns with transaction costs
    strategy_returns = signals[:-1] * returns * 0.998  # 0.2% transaction cost per trade
    cumulative_returns = (1 + strategy_returns).cumprod()

    # buy and hold benchmark
    buy_hold_returns = (prices[-1] / prices[0]) - 1

    # performance metrics
    total_return = cumulative_returns[-1] - 1
    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252 * 12 * 24)  # annualized
    max_drawdown = calculate_max_drawdown(cumulative_returns)

    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_returns,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'cumulative_returns': cumulative_returns,
        'signals': signals
    }


def calculate_max_drawdown(cumulative_returns):
    """calculating maximum drawdown of strategy"""
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return np.min(drawdown)


def create_performance_visualization(results, train_losses, val_accuracies):
    """
    creating     performance visualization plots
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # training curves
    ax1.plot(train_losses, 'b-', label='training loss', linewidth=2)
    ax1.set_title('model training loss', fontweight='bold')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(val_accuracies, 'g-', label='validation accuracy', linewidth=2)
    ax2.set_title('model validation accuracy', fontweight='bold')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # cumulative returns comparison
    ax3.plot(results['cumulative_returns'], 'r-', label='lstm strategy', linewidth=2)
    ax3.axhline(y=1 + results['buy_hold_return'], color='b', linestyle='--',
                label='buy & hold', linewidth=2)
    ax3.set_title('cumulative returns comparison', fontweight='bold')
    ax3.set_xlabel('time periods')
    ax3.set_ylabel('cumulative return')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # performance metrics bar chart
    metrics = ['strategy return', 'buy & hold return', 'sharpe ratio', 'max drawdown']
    values = [results['total_return'], results['buy_hold_return'],
              results['sharpe_ratio'], abs(results['max_drawdown'])]
    colors = ['green' if v > 0 else 'red' for v in values]

    bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_title('performance metrics', fontweight='bold')
    ax4.set_ylabel('value')
    ax4.grid(True, alpha=0.3)

    # adding value labels on bars  ***
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01 if height > 0 else height - 0.01,
                 f'{value:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                 fontweight='bold')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('microstructure_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("performance visualization saved as 'microstructure_performance.png'")


def main():
    """main execution function"""

    print("starting high-frequency microstructure lstm analysis")

    # fetching and preparing data
    raw_data = get_spy_data()
    if raw_data is None:
        print("failed to fetch data. exiting.")
        return

    # simulating microstructure features
    sim_df = simulate_microstructure_data(raw_data)

    # preparing training data
    features, targets, directions, scaler_feat, scaler_targ = prepare_training_data(sim_df)

    # combining targets for multi-task learning
    combined_targets = np.column_stack((targets, directions))

    # creating train/val split (80/20)
    split_idx = int(len(features) * 0.8)
    train_features, val_features = features[:split_idx], features[split_idx:]
    train_targets, val_targets = combined_targets[:split_idx], combined_targets[split_idx:]

    # creating datasets and dataloaders
    sequence_length = 20
    train_dataset = MicrostructureDataset(train_features, train_targets, sequence_length)
    val_dataset = MicrostructureDataset(val_features, val_targets, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # initializing and train model
    input_size = features.shape[1]
    model = MicrostructureLSTM(input_size=input_size, hidden_size=64, num_layers=2)

    print("training microstructure lstm model...")
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, num_epochs=25)

    # backtesting trading strategy
    print("backtesting trading strategy...")
    results = backtest_strategy(sim_df, model, scaler_feat, scaler_targ, sequence_length)

    # printing results
    print(f"\n***--- performance results ---***")
    print(f"strategy total return: {results['total_return']:.2%}")
    print(f"buy & hold return: {results['buy_hold_return']:.2%}")
    print(f"sharpe ratio: {results['sharpe_ratio']:.3f}")
    print(f"maximum drawdown: {results['max_drawdown']:.2%}")
    print(f"final validation accuracy: {val_accuracies[-1]:.2%}")

    #
    create_performance_visualization(results, train_losses, val_accuracies)

    print("\nanalysis complete!")


if __name__ == "__main__":
    main()