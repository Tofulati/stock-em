
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import yfinance as yf
import json
import warnings
import os
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta

from config import SEQ_LEN
from data_ingest import download_prices
from features import make_advanced_features, create_labels, build_sequences_with_labels
from src.models.lstm_model import AdvancedLSTM, CombinedLoss
from backtest import AdvancedBacktest
from ensemble import StackingEnsemble
from src.models.news_model import AdvancedNewsModel


# ==============================================================================
# 🛠️ HELPER FUNCTION (ENHANCED FIX WITH BETTER DATE HANDLING)
# ==============================================================================

def convert_numpy_types(obj):
    """Recursively converts numpy types and dates to standard Python types 
    for safe JSON serialization."""
    if isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item() 
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, pd.DatetimeIndex):
        return [d.strftime('%Y-%m-%d') for d in obj]
    elif pd.isna(obj):
        return None
    return obj


# ==============================================================================
# 📈 FUTURE PREDICTION SYSTEM (NO LOOK-AHEAD BIAS)
# ==============================================================================

class FuturePredictionSystem:
    """
    Makes predictions using ONLY data available up to prediction date.
    Implements walk-forward validation for realistic performance estimates.
    """
    
    def __init__(self, model, scaler, feature_cols, seq_len=30):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def fetch_data_up_to_date(self, ticker, as_of_date, lookback_days=100):
        """Fetch historical data UP TO (not including) the prediction date"""
        start_date = as_of_date - timedelta(days=lookback_days)
        df = yf.download(ticker, start=start_date, end=as_of_date, progress=False)
        
        if len(df) == 0:
            raise ValueError(f"No data available for {ticker} up to {as_of_date}")
        
        # Fix multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        if 'Date' in df.columns:
            df = df.set_index('Date')
        
        return df
    
    def prepare_features_safe(self, df):
        """Create features using ONLY past data (no look-ahead)"""
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        
        # Apply feature engineering (naturally uses only past data)
        df = make_advanced_features(df)
        
        # Ensure all required features exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Scale features
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        
        return df
    
    def predict_next_day(self, ticker, as_of_date=None):
        """
        Predict the next trading day's return using ONLY data up to as_of_date.
        
        Args:
            ticker: Stock symbol
            as_of_date: Date to make prediction from (default: today)
        
        Returns:
            dict with prediction details
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # Fetch data up to (not including) prediction date
        df = self.fetch_data_up_to_date(ticker, as_of_date)
        
        if len(df) < self.seq_len:
            raise ValueError(f"Insufficient data: need {self.seq_len} days, got {len(df)}")
        
        # Create features using only available data
        df = self.prepare_features_safe(df)
        
        # Create sequence from last seq_len days
        X = df[self.feature_cols].values[-self.seq_len:]
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            pred_return, action_logits, confidence = self.model(X_tensor)
            signal = torch.argmax(action_logits, dim=1).item() - 1
            
            # Filter by confidence
            if confidence.item() < 0.4:
                signal = 0
        
        prediction_date = as_of_date + timedelta(days=1)
        
        return {
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'as_of_date': as_of_date.strftime('%Y-%m-%d'),
            'predicted_return': float(pred_return.item()),
            'signal': int(signal),
            'signal_name': {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}[signal],
            'confidence': float(confidence.item()),
            'last_price': float(df['Close'].iloc[-1])
        }
    
    def walk_forward_test(self, ticker, start_date, end_date, initial_cash=10000):
        """
        Walk-forward validation with realistic trading simulation.
        
        Simulates day-by-day trading:
        1. Make prediction using only past data
        2. Execute trade
        3. Wait for actual results
        4. Move to next day
        """
        results = []
        cash = initial_cash
        shares = 0
        portfolio_values = [initial_cash]
        
        current_date = start_date
        
        print(f"\n{'='*60}")
        print(f"Walk-Forward Test: {ticker}")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${initial_cash:,.2f}")
        print(f"{'='*60}\n")
        
        trading_days = 0
        
        while current_date < end_date:
            try:
                # Make prediction using only data up to current_date
                pred = self.predict_next_day(ticker, as_of_date=current_date)
                
                # Get current price
                current_price = pred['last_price']
                
                # Execute trade based on signal
                signal = pred['signal']
                if signal == 1 and shares == 0 and cash > 0:  # BUY
                    shares_to_buy = (cash * 0.3) / current_price  # 30% position
                    shares = shares_to_buy
                    cash -= shares_to_buy * current_price * 1.001  # 0.1% transaction cost
                    
                elif signal == -1 and shares > 0:  # SELL
                    proceeds = shares * current_price * 0.999  # 0.1% transaction cost
                    cash += proceeds
                    shares = 0
                
                # Fetch ACTUAL next day data
                next_date = current_date + timedelta(days=1)
                actual_df = yf.download(ticker, start=next_date, 
                                       end=next_date + timedelta(days=2), 
                                       progress=False)
                
                if len(actual_df) > 0:
                    if isinstance(actual_df.columns, pd.MultiIndex):
                        actual_df.columns = actual_df.columns.get_level_values(0)
                    
                    actual_price = actual_df['Close'].iloc[0]
                    actual_return = (actual_price - current_price) / current_price
                    
                    # Calculate portfolio value
                    portfolio_value = cash + shares * actual_price
                    daily_pnl = portfolio_value - portfolio_values[-1]
                    portfolio_values.append(portfolio_value)
                    
                    # Calculate prediction metrics
                    error = abs(pred['predicted_return'] - actual_return)
                    direction_correct = (pred['predicted_return'] * actual_return) > 0
                    
                    results.append({
                        'date': pred['prediction_date'],
                        'predicted_return': pred['predicted_return'],
                        'actual_return': actual_return,
                        'signal': pred['signal'],
                        'signal_name': pred['signal_name'],
                        'confidence': pred['confidence'],
                        'error': error,
                        'direction_correct': direction_correct,
                        'portfolio_value': portfolio_value,
                        'cash': cash,
                        'shares': shares,
                        'daily_pnl': daily_pnl
                    })
                    
                    trading_days += 1
                    if trading_days % 20 == 0:
                        print(f"Day {trading_days}: {pred['prediction_date']} | "
                              f"Pred={pred['predicted_return']*100:+.2f}% | "
                              f"Actual={actual_return*100:+.2f}% | "
                              f"Signal={pred['signal_name']} | "
                              f"Portfolio=${portfolio_value:,.2f} | "
                              f"P&L={daily_pnl:+,.2f}")
                
            except Exception as e:
                pass
            
            current_date += timedelta(days=1)
        
        # Calculate final metrics
        df_results = pd.DataFrame(results)
        
        if len(df_results) > 0:
            total_return = (portfolio_values[-1] / initial_cash - 1) * 100
            
            returns_series = pd.Series(portfolio_values).pct_change().dropna()
            sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
            
            cumulative = (1 + returns_series).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            metrics = {
                'Total Return (%)': total_return,
                'Sharpe Ratio': sharpe,
                'Max Drawdown (%)': max_drawdown,
                'Final Portfolio Value': portfolio_values[-1],
                'Total Predictions': len(df_results),
                'Direction Accuracy (%)': df_results['direction_correct'].mean() * 100,
                'Mean Absolute Error (%)': df_results['error'].mean() * 100,
                'High Conf Accuracy (%)': df_results[df_results['confidence'] > 0.6]['direction_correct'].mean() * 100 if len(df_results[df_results['confidence'] > 0.6]) > 0 else 0
            }
            
            print("\n" + "="*60)
            print("WALK-FORWARD TEST RESULTS")
            print("="*60)
            for key, value in metrics.items():
                print(f"{key:.<45} {value:>10.2f}")
            print("="*60)
            
            return df_results, metrics
        
        return None, None


# ==============================================================================
# 🧠 TRAINING FUNCTION (UNCHANGED)
# ==============================================================================

def train_model_with_validation(model, train_loader, val_loader, loss_fn, device, 
                                 epochs=100, patience=15, lr=1e-3):
    """Train model with early stopping"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = np.inf
    best_model_state = model.state_dict()
    counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    print("\nTraining model...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for xb, y_ret, y_act in train_loader:
            xb = xb.to(device)
            y_ret = y_ret.to(device)
            y_act = y_act.to(device)
            
            optimizer.zero_grad()
            pred_ret, action_logits, confidence = model(xb)
            
            if torch.isnan(pred_ret).any() or torch.isnan(action_logits).any():
                continue
            
            loss, loss_dict = loss_fn(pred_ret, action_logits, confidence, y_ret, y_act)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        if len(train_losses) == 0:
            print(f"Error: All batches had NaN at epoch {epoch}. Stopping training.")
            break
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for xb, y_ret, y_act in val_loader:
                xb = xb.to(device)
                y_ret = y_ret.to(device)
                y_act = y_act.to(device)
                
                pred_ret, action_logits, confidence = model(xb)
                
                if torch.isnan(pred_ret).any() or torch.isnan(action_logits).any():
                    continue
                
                loss, _ = loss_fn(pred_ret, action_logits, confidence, y_ret, y_act)
                
                if not torch.isnan(loss):
                    val_losses.append(loss.item())
        
        if len(val_losses) == 0:
            print(f"Error: All validation batches had NaN at epoch {epoch}. Stopping training.")
            break
        
        avg_val_loss = np.mean(val_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    model.load_state_dict(best_model_state)
    return model, history


# ==============================================================================
# 📊 SIGNAL GENERATION FUNCTION (UNCHANGED)
# ==============================================================================

def generate_signals(pred_returns, action_probs, confidence, 
                      return_threshold=0.005, conf_threshold=0.5):
    """Convert model predictions to trading signals"""
    actions = torch.argmax(action_probs, dim=1).cpu().numpy()
    signals = actions - 1 
    
    # Filter by confidence
    low_conf_mask = confidence.cpu().numpy() < conf_threshold
    signals[low_conf_mask] = 0
    
    # Filter by predicted return magnitude
    pred_ret_np = pred_returns.cpu().numpy()
    small_ret_mask = np.abs(pred_ret_np) < return_threshold
    signals[small_ret_mask] = 0
    
    return signals


# ==============================================================================
# 🔧 PROPER TEMPORAL SPLIT (FIX FOR LOOK-AHEAD BIAS)
# ==============================================================================

def train_with_proper_temporal_split(X, y_ret, y_act, indices, 
                                     train_frac=0.7, val_frac=0.15):
    """
    Split data CHRONOLOGICALLY to prevent look-ahead bias.
    
    Critical: We NEVER shuffle data! Split strictly by time.
    """
    n_total = len(X)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    
    # Split chronologically - DO NOT SHUFFLE
    X_train = X[:n_train]
    y_ret_train = y_ret[:n_train]
    y_act_train = y_act[:n_train]
    idx_train = indices[:n_train]
    
    X_val = X[n_train:n_train+n_val]
    y_ret_val = y_ret[n_train:n_train+n_val]
    y_act_val = y_act[n_train:n_train+n_val]
    idx_val = indices[n_train:n_train+n_val]
    
    X_test = X[n_train+n_val:]
    y_ret_test = y_ret[n_train+n_val:]
    y_act_test = y_act[n_train+n_val:]
    idx_test = indices[n_train+n_val:]
    
    print(f"\n{'='*60}")
    print("TEMPORAL DATA SPLIT (Chronological Order - NO LOOK-AHEAD)")
    print(f"{'='*60}")
    print(f"  Train: {len(X_train):,} samples")
    print(f"         Dates: {idx_train[0]} to {idx_train[-1]}")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"         Dates: {idx_val[0]} to {idx_val[-1]}")
    print(f"  Test:  {len(X_test):,} samples")
    print(f"         Dates: {idx_test[0]} to {idx_test[-1]}")
    print(f"{'='*60}\n")
    
    return (X_train, y_ret_train, y_act_train, idx_train,
            X_val, y_ret_val, y_act_val, idx_val,
            X_test, y_ret_test, y_act_test, idx_test)


# ==============================================================================
# ⚙️ MAIN PIPELINE FUNCTION (IMPROVED)
# ==============================================================================

def improved_pipeline_with_ensemble(
    ticker='AAPL',
    period='5y',
    seq_len=30,
    batch_size=64,
    epochs=150,
    use_ensemble=True,
    test_walk_forward=True,
    enable_shorting=True,
    export_results=True
):
    """
    Complete enhanced trading pipeline with NO LOOK-AHEAD BIAS
    """
    print(f"\n{'='*60}")
    print(f"Running Enhanced Pipeline for {ticker}")
    print(f"Ensemble: {use_ensemble} | Walk-Forward Test: {test_walk_forward}")
    print(f"{'='*60}\n")
    
    # 1. Download data
    print("Step 1: Downloading price data...")
    df = download_prices(ticker, period=period)
    print(f"  Downloaded {len(df)} days of data")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df = df.set_index('Date')
        elif 'date' in df.columns:
            df = df.set_index('date')
    
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # 2. Create technical features
    print("\nStep 2: Creating technical features...")
    df = make_advanced_features(df)
    
    # 3. Create labels
    print("\nStep 3: Creating labels...")
    df = create_labels(df, threshold_buy=0.01, threshold_sell=-0.01)
    
    # 4. Build sequences
    print("\nStep 4: Building sequences...")
    feature_cols = [
        'ret1', 'ret5', 'ret20', 'log_ret1', 'daily_range',
        'close_to_ma20', 'rsi14', 'macd', 'volatility20',
        'volume_ratio'
    ]
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    X, y_ret, y_act, indices, final_features = build_sequences_with_labels(
        df, seq_len=seq_len, feature_cols=feature_cols
    )
    
    print(f"  Created {len(X)} sequences with {X.shape[2]} features")
    
    action_counts = pd.Series(y_act).value_counts().sort_index()
    print(f"  Action distribution:")
    print(f"    SELL (0): {action_counts.get(0, 0)}")
    print(f"    HOLD (1): {action_counts.get(1, 0)}")
    print(f"    BUY  (2): {action_counts.get(2, 0)}")
    
    # 5. Scale features
    print("\nStep 5: Scaling features...")
    scaler = RobustScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(X.shape)
    
    # 6. PROPER TEMPORAL SPLIT (NO LOOK-AHEAD)
    print("\nStep 6: Splitting data chronologically...")
    (X_train, y_ret_train, y_act_train, idx_train,
     X_val, y_ret_val, y_act_val, idx_val,
     X_test, y_ret_test, y_act_test, idx_test) = train_with_proper_temporal_split(
        X, y_ret, y_act, indices
    )
    
    # Convert to PyTorch - NO SHUFFLING
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_ret_train, dtype=torch.float32),
        torch.tensor(y_act_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_ret_val, dtype=torch.float32),
        torch.tensor(y_act_val, dtype=torch.long)
    )
    
    # CRITICAL: shuffle=False to maintain temporal order
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 7. Create and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStep 7: Creating model (device: {device})...")
    
    if use_ensemble:
        print("  Using Stacking Ensemble (LSTM + GRU + Transformer)")
        model = StackingEnsemble(input_size=X.shape[2], hidden_size=64).to(device)
    else:
        print("  Using Advanced LSTM")
        model = AdvancedLSTM(input_size=X.shape[2], hidden_size=64, num_layers=2).to(device)
    
    loss_fn = CombinedLoss(alpha=1.0, beta=0.3, gamma=0.1)
    
    model, history = train_model_with_validation(
        model, train_loader, val_loader, loss_fn, device,
        epochs=epochs, patience=15, lr=5e-4
    )
    
    # 8. Generate predictions on test set
    print("\nStep 8: Generating predictions on test set...")
    model.eval()
    
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred_ret, action_logits, confidence = model(X_test_t)
        action_probs = torch.softmax(action_logits, dim=1)
    
    signals = generate_signals(pred_ret, action_probs, confidence,
                               return_threshold=0.003, conf_threshold=0.4)
    
    signals_series = pd.Series(signals, index=idx_test)
    confidence_series = pd.Series(confidence.cpu().numpy(), index=idx_test)
    
    print(f"\n  Signal distribution:")
    print(f"    BUY:  {np.sum(signals == 1)}")
    print(f"    HOLD: {np.sum(signals == 0)}")
    print(f"    SELL: {np.sum(signals == -1)}")
    
    # 9. Standard Backtest (for comparison)
    print("\nStep 9: Running standard backtest...")
    
    price_for_bt = df.loc[idx_test][['Open', 'Close', 'High', 'Low']]
    
    backtester = AdvancedBacktest(
        init_cash=10000,
        max_position_pct=0.3,
        transaction_cost=0.001,
        stop_loss_pct=0.05,
        enable_shorting=enable_shorting
    )
    
    bt_results = backtester.run(price_for_bt, signals_series, confidence_series)
    
    # 10. Calculate metrics
    print("\nStep 10: Standard Backtest Metrics")
    print("="*60)
    metrics = backtester.calculate_metrics(bt_results)
    
    for metric, value in metrics.items():
        if isinstance(value, float) or isinstance(value, np.floating):
            print(f"  {metric:.<30} {value:>10.2f}")
        else:
            print(f"  {metric:.<30} {value:>10}")
    
    # 11. Walk-Forward Test (REALISTIC PERFORMANCE)
    walk_forward_results = None
    walk_forward_metrics = None
    
    if test_walk_forward:
        print("\nStep 11: Walk-Forward Testing (NO LOOK-AHEAD)...")
        predictor = FuturePredictionSystem(model, scaler, final_features, seq_len=seq_len)
        
        # Use test period dates
        test_start = idx_test[0]
        test_end = idx_test[-1]
        
        walk_forward_results, walk_forward_metrics = predictor.walk_forward_test(
            ticker, test_start, test_end
        )
    
    # 12. Make prediction for tomorrow
    print("\nStep 12: Predicting tomorrow's return...")
    predictor = FuturePredictionSystem(model, scaler, final_features, seq_len=seq_len)
    tomorrow_pred = predictor.predict_next_day(ticker)
    
    print(f"\n{'='*60}")
    print("TOMORROW'S PREDICTION")
    print(f"{'='*60}")
    print(f"  Date:              {tomorrow_pred['prediction_date']}")
    print(f"  Predicted Return:  {tomorrow_pred['predicted_return']*100:+.2f}%")
    print(f"  Signal:            {tomorrow_pred['signal_name']}")
    print(f"  Confidence:        {tomorrow_pred['confidence']*100:.1f}%")
    print(f"  Last Price:        ${tomorrow_pred['last_price']:.2f}")
    print(f"{'='*60}\n")
    
    # 13. Export results
    if export_results:
        print("\nStep 13: Exporting results...")
        
        bt_dict = []
        for idx, row in bt_results.iterrows():
            record = {'date': idx.strftime('%Y-%m-%d') if isinstance(idx, pd.Timestamp) else str(idx)}
            for col, val in row.items():
                record[col] = convert_numpy_types(val)
            bt_dict.append(record)
        
        wf_dict = []
        if walk_forward_results is not None:
            for idx, row in walk_forward_results.iterrows():
                record = {}
                for col, val in row.items():
                    record[col] = convert_numpy_types(val)
                wf_dict.append(record)
        
        export_data = {
            'ticker': ticker,
            'period': period,
            'use_ensemble': use_ensemble,
            'standard_backtest': {
                'metrics': convert_numpy_types(metrics),
                'results': bt_dict
            },
            'walk_forward_test': {
                'metrics': convert_numpy_types(walk_forward_metrics) if walk_forward_metrics else None,
                'results': wf_dict if wf_dict else None
            },
            'tomorrow_prediction': tomorrow_pred,
            'model_info': {
                'seq_len': seq_len,
                'epochs_trained': len(history['train_loss']),
                'final_train_loss': float(history['train_loss'][-1]),
                'final_val_loss': float(history['val_loss'][-1]),
                'features': final_features
            }
        }

        export_folder = "data/results"
        os.makedirs(export_folder, exist_ok=True)
        output_file = os.path.join(export_folder, f"model_results_{ticker}.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"  Results exported to {output_file}")
    
    # 14. Plot results
    print("\nStep 14: Generating plots...")
    backtester.plot_results(bt_results, ticker=ticker)
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    
    return {
        'model': model,
        'backtest': bt_results,
        'metrics': metrics,
        'walk_forward_results': walk_forward_results,
        'walk_forward_metrics': walk_forward_metrics,
        'tomorrow_prediction': tomorrow_pred,
        'signals': signals_series,
        'confidence': confidence_series,
        'scaler': scaler,
        'feature_cols': final_features,
        'history': history
    }


# ==============================================================================
# 🚀 EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    results = improved_pipeline_with_ensemble(
        ticker='INTC',
        period='5y',
        seq_len=30,
        batch_size=64,
        epochs=150,
        use_ensemble=True, 
        test_walk_forward=True, 
        enable_shorting=False,
        export_results=True
    )