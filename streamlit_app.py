# streamlit web app for high-frequency microstructure lstm project
# deployment: streamlit run streamlit_app.py
# cache the trained model to avoid retraining on each interaction

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
import time
import io

warnings.filterwarnings('ignore')

# import all functions from main.py
from main import (
    simulate_microstructure_data, get_spy_data, MicrostructureDataset,
    MicrostructureLSTM, prepare_training_data, train_model, backtest_strategy,
    calculate_max_drawdown
)

# set page config
st.set_page_config(
    page_title="Microstructure LSTM Trading",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# cache functions for performance
@st.cache_data
def load_spy_data():
    """load spy data with caching"""
    return get_spy_data()

@st.cache_data
def simulate_microstructure_cached(data_hash):
    """simulate microstructure data with caching based on data hash"""
    raw_data = load_spy_data()
    if raw_data is not None:
        return simulate_microstructure_data(raw_data)
    return None

@st.cache_resource
def train_cached_model(_sim_df, epochs, sequence_length, batch_size, hidden_size):
    """train and cache the lstm model to avoid retraining"""
    try:
        # prepare training data
        features, targets, directions, scaler_feat, scaler_targ = prepare_training_data(_sim_df, sequence_length)
        
        # combine targets for multi-task learning
        combined_targets = np.column_stack((targets, directions))
        
        # train/val split
        split_idx = int(len(features) * 0.8)
        train_features, val_features = features[:split_idx], features[split_idx:]
        train_targets, val_targets = combined_targets[:split_idx], combined_targets[split_idx:]
        
        # create datasets
        train_dataset = MicrostructureDataset(train_features, train_targets, sequence_length)
        val_dataset = MicrostructureDataset(val_features, val_targets, sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # create model
        input_size = features.shape[1]
        model = MicrostructureLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2)
        
        # train model
        train_losses, val_accuracies = train_model(model, train_loader, val_loader, num_epochs=epochs)
        
        return model, train_losses, val_accuracies, scaler_feat, scaler_targ, features, targets, directions
        
    except Exception as e:
        st.error(f"error training model: {e}")
        return None, None, None, None, None, None, None, None

def create_plotly_charts(sim_df, train_losses, val_accuracies, results):
    """create interactive plotly charts for the dashboard"""
    
    # training loss chart
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(
        x=list(range(len(train_losses))),
        y=train_losses,
        mode='lines',
        name='training loss',
        line=dict(color='blue', width=2)
    ))
    loss_fig.update_layout(
        title='model training loss',
        xaxis_title='epoch',
        yaxis_title='loss',
        height=400
    )
    
    # validation accuracy chart
    acc_fig = go.Figure()
    acc_fig.add_trace(go.Scatter(
        x=list(range(len(val_accuracies))),
        y=val_accuracies,
        mode='lines',
        name='validation accuracy',
        line=dict(color='green', width=2)
    ))
    acc_fig.update_layout(
        title='model validation accuracy',
        xaxis_title='epoch',
        yaxis_title='accuracy',
        height=400
    )
    
    # cumulative returns comparison
    returns_fig = go.Figure()
    returns_fig.add_trace(go.Scatter(
        x=list(range(len(results['cumulative_returns']))),
        y=results['cumulative_returns'],
        mode='lines',
        name='lstm strategy',
        line=dict(color='red', width=2)
    ))
    returns_fig.add_hline(
        y=1 + results['buy_hold_return'],
        line_dash="dash",
        line_color="blue",
        annotation_text="buy & hold"
    )
    returns_fig.update_layout(
        title='cumulative returns comparison',
        xaxis_title='time periods',
        yaxis_title='cumulative return',
        height=400
    )
    
    # performance metrics bar chart
    metrics_fig = go.Figure()
    metrics = ['strategy return', 'buy & hold', 'sharpe ratio', 'max drawdown']
    values = [
        results['total_return'],
        results['buy_hold_return'],
        results['sharpe_ratio'],
        abs(results['max_drawdown'])
    ]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    metrics_fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker_color=colors,
        text=[f'{v:.3f}' for v in values],
        textposition='auto'
    ))
    metrics_fig.update_layout(
        title='performance metrics',
        yaxis_title='value',
        height=400
    )
    
    return loss_fig, acc_fig, returns_fig, metrics_fig

def create_microstructure_charts(sim_df):
    """create charts for microstructure data visualization"""
    
    # bid-ask spread over time
    spread_fig = go.Figure()
    spread_fig.add_trace(go.Scatter(
        x=sim_df.index,
        y=sim_df['bid_ask_spread'],
        mode='lines',
        name='bid-ask spread',
        line=dict(color='purple', width=1)
    ))
    spread_fig.update_layout(
        title='bid-ask spread over time',
        xaxis_title='time',
        yaxis_title='spread',
        height=300
    )
    
    # order flow imbalance
    flow_fig = go.Figure()
    flow_fig.add_trace(go.Scatter(
        x=sim_df.index,
        y=sim_df['order_flow_imbalance'],
        mode='lines',
        name='order flow imbalance',
        line=dict(color='orange', width=1)
    ))
    flow_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    flow_fig.update_layout(
        title='order flow imbalance (buying vs selling pressure)',
        xaxis_title='time',
        yaxis_title='order flow',
        height=300
    )
    
    # volatility
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Scatter(
        x=sim_df.index,
        y=sim_df['volatility'],
        mode='lines',
        name='volatility',
        line=dict(color='red', width=1)
    ))
    vol_fig.update_layout(
        title='price volatility',
        xaxis_title='time',
        yaxis_title='volatility',
        height=300
    )
    
    return spread_fig, flow_fig, vol_fig

def main():
    """main streamlit app"""
    
    st.title("high-frequency microstructure lstm trading")
    st.markdown("interactive demo of lstm model predicting bid-ask spreads and trading spy")
    
    # sidebar configuration
    st.sidebar.header("model configuration")
    epochs = st.sidebar.slider("training epochs", min_value=5, max_value=50, value=20, step=5)
    sequence_length = st.sidebar.slider("sequence length", min_value=10, max_value=50, value=20, step=5)
    batch_size = st.sidebar.selectbox("batch size", [16, 32, 64, 128], index=1)
    hidden_size = st.sidebar.selectbox("hidden size", [32, 64, 128, 256], index=1)
    
    # load data button
    if st.sidebar.button("load fresh data & train model", type="primary"):
        st.cache_data.clear()
        st.cache_resource.clear()
    
    # load and simulate data
    with st.spinner("loading spy data..."):
        raw_data = load_spy_data()
    
    if raw_data is None:
        st.error("failed to load spy data. check internet connection.")
        return
    
    with st.spinner("simulating microstructure features..."):
        # create a simple hash for caching
        data_hash = hash(str(raw_data.shape) + str(raw_data.iloc[0]['close']))
        sim_df = simulate_microstructure_cached(data_hash)
    
    if sim_df is None:
        st.error("failed to simulate microstructure data.")
        return
    
    st.success(f"loaded {len(sim_df)} periods of microstructure data")
    
    # create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["data & preprocessing", "model training", "trading performance", "predictions"])
    
    # tab 1: data & preprocessing
    with tab1:
        st.header("microstructure data simulation")
        st.markdown("simulated bid-ask spreads and order flow from real spy price/volume data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("raw spy data sample")
            st.dataframe(raw_data.head(10), use_container_width=True)
            
            st.subheader("feature statistics")
            feature_stats = sim_df[['bid_ask_spread', 'order_flow_imbalance', 'volatility', 'volume_ratio']].describe()
            st.dataframe(feature_stats, use_container_width=True)
        
        with col2:
            st.subheader("simulated microstructure features")
            st.dataframe(sim_df[['close', 'bid_ask_spread', 'order_flow_imbalance', 'volatility', 'volume_ratio']].head(10), use_container_width=True)
        
        # microstructure visualizations
        st.subheader("microstructure visualizations")
        spread_fig, flow_fig, vol_fig = create_microstructure_charts(sim_df)
        
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(spread_fig, use_container_width=True)
            st.plotly_chart(vol_fig, use_container_width=True)
        with col4:
            st.plotly_chart(flow_fig, use_container_width=True)
    
    # tab 2: model training
    with tab2:
        st.header("lstm model training")
        
        # train model with progress bar
        if st.button("train model", type="primary", key="train_button"):
            with st.spinner(f"training lstm model for {epochs} epochs..."):
                progress_bar = st.progress(0)
                start_time = time.time()
                
                # simulate training progress
                for i in range(epochs):
                    time.sleep(0.1)  # small delay to show progress
                    progress_bar.progress((i + 1) / epochs)
                
                # actual training
                model_results = train_cached_model(sim_df, epochs, sequence_length, batch_size, hidden_size)
                model, train_losses, val_accuracies, scaler_feat, scaler_targ, features, targets, directions = model_results
                
                training_time = time.time() - start_time
                
                if model is not None:
                    st.success(f"model trained successfully in {training_time:.1f} seconds!")
                    
                    # store in session state
                    st.session_state.model = model
                    st.session_state.train_losses = train_losses
                    st.session_state.val_accuracies = val_accuracies
                    st.session_state.scaler_feat = scaler_feat
                    st.session_state.scaler_targ = scaler_targ
                    
                    # training curves
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        loss_fig = go.Figure()
                        loss_fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='training loss'))
                        loss_fig.update_layout(title='training loss curve', height=400)
                        st.plotly_chart(loss_fig, use_container_width=True)
                    
                    with col2:
                        acc_fig = go.Figure()
                        acc_fig.add_trace(go.Scatter(y=val_accuracies, mode='lines', name='validation accuracy'))
                        acc_fig.update_layout(title='validation accuracy curve', height=400)
                        st.plotly_chart(acc_fig, use_container_width=True)
                    
                    # model metrics
                    col3, col4, col5 = st.columns(3)
                    with col3:
                        st.metric("final training loss", f"{train_losses[-1]:.4f}")
                    with col4:
                        st.metric("final validation accuracy", f"{val_accuracies[-1]:.2%}")
                    with col5:
                        st.metric("training time", f"{training_time:.1f}s")
                else:
                    st.error("model training failed!")
        else:
            # try to load from session state or cache
            try:
                model_results = train_cached_model(sim_df, epochs, sequence_length, batch_size, hidden_size)
                model, train_losses, val_accuracies, scaler_feat, scaler_targ, features, targets, directions = model_results
                
                if model is not None:
                    st.session_state.model = model
                    st.session_state.train_losses = train_losses
                    st.session_state.val_accuracies = val_accuracies
                    st.session_state.scaler_feat = scaler_feat
                    st.session_state.scaler_targ = scaler_targ
                    
                    st.info("using cached trained model. click 'train model' to retrain.")
            except:
                st.info("click 'train model' to start training the lstm model.")
    
    # tab 3: trading performance
    with tab3:
        st.header("trading strategy performance")
        
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            with st.spinner("backtesting trading strategy..."):
                results = backtest_strategy(
                    sim_df, 
                    st.session_state.model, 
                    st.session_state.scaler_feat, 
                    st.session_state.scaler_targ, 
                    sequence_length
                )
            
            # performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "strategy return",
                    f"{results['total_return']:.2%}",
                    delta=f"{results['total_return'] - results['buy_hold_return']:.2%}"
                )
            with col2:
                st.metric("buy & hold return", f"{results['buy_hold_return']:.2%}")
            with col3:
                st.metric("sharpe ratio", f"{results['sharpe_ratio']:.3f}")
            with col4:
                st.metric("max drawdown", f"{results['max_drawdown']:.2%}")
            
            # charts
            if hasattr(st.session_state, 'train_losses'):
                loss_fig, acc_fig, returns_fig, metrics_fig = create_plotly_charts(
                    sim_df, st.session_state.train_losses, st.session_state.val_accuracies, results
                )
                
                col5, col6 = st.columns(2)
                with col5:
                    st.plotly_chart(returns_fig, use_container_width=True)
                with col6:
                    st.plotly_chart(metrics_fig, use_container_width=True)
            
            # download predictions
            if st.button("download predictions csv"):
                predictions_df = pd.DataFrame({
                    'signal': results['signals'],
                    'cumulative_return': results['cumulative_returns']
                })
                
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="download csv file",
                    data=csv,
                    file_name="lstm_predictions.csv",
                    mime="text/csv"
                )
        else:
            st.warning("train the model first in the 'model training' tab!")
    
    # tab 4: predictions
    with tab4:
        st.header("sample predictions with confidence")
        
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            st.subheader("recent predictions")
            
            # make predictions on recent data
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = st.session_state.model.to(device)
            model.eval()
            
            feature_cols = ['volume_ratio', 'price_momentum', 'volatility', 'order_flow_imbalance']
            test_features = st.session_state.scaler_feat.transform(sim_df[feature_cols])
            
            recent_predictions = []
            
            with torch.no_grad():
                for i in range(max(0, len(test_features) - 50), len(test_features) - sequence_length):
                    seq = test_features[i:i + sequence_length]
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
                    
                    spread_pred, direction_pred = model(seq_tensor)
                    direction_prob = torch.softmax(direction_pred, dim=1)
                    
                    confidence = direction_prob[0, 1].item()
                    signal = 1 if confidence > 0.6 else (-1 if confidence < 0.4 else 0)
                    
                    recent_predictions.append({
                        'time_idx': i + sequence_length,
                        'predicted_spread': spread_pred.item(),
                        'direction_confidence': confidence,
                        'trading_signal': signal,
                        'signal_text': 'long' if signal == 1 else ('short' if signal == -1 else 'hold')
                    })
            
            # display predictions table
            pred_df = pd.DataFrame(recent_predictions)
            st.dataframe(pred_df, use_container_width=True)
            
            # prediction confidence distribution
            conf_fig = go.Figure(data=[go.Histogram(x=pred_df['direction_confidence'], nbinsx=20)])
            conf_fig.update_layout(
                title='prediction confidence distribution',
                xaxis_title='confidence score',
                yaxis_title='frequency',
                height=400
            )
            st.plotly_chart(conf_fig, use_container_width=True)
            
        else:
            st.warning("train the model first in the 'model training' tab!")

if __name__ == "__main__":
    main()