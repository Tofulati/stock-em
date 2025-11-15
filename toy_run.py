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
# 🛠️ HELPER FUNCTION (THE FIX)
# ==============================================================================

def convert_numpy_types(obj):
    """Recursively converts numpy types (like int64, float64) to standard Python types 
    for safe JSON serialization."""
    if isinstance(obj, dict):
        # Convert dictionary keys/values
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.floating)):
        # Convert NumPy integer/float to standard Python int/float
        return obj.item() 
    elif isinstance(obj, (list, tuple)):
        # Convert list/tuple elements
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        # Convert datetime objects to string format
        return obj.isoformat()
    return obj


# ==============================================================================
# 📈 LIVE TESTER CLASS
# ==============================================================================

class LiveTester:
    """Test model on live data with walk-forward validation"""
    def __init__(self, model, scaler, feature_cols):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cash = 10000
        self.shares = 0
        
    def fetch_recent_data(self, ticker, end_date, days=100):
        """Fetch data up to specific date"""
        start_date = end_date - timedelta(days=days)
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Fix multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to make date a column, then set it back
        df = df.reset_index()
        if 'Date' in df.columns:
            df = df.set_index('Date')
        
        return df
    
    def prepare_features(self, df):
        """Apply feature engineering"""
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure columns are strings
        df.columns = [str(c) for c in df.columns]
        
        # Apply feature engineering
        df = make_advanced_features(df)
        
        # Ensure all required features exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Scale features
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df
    
    def create_sequence(self, df, seq_len=30):
        """Create sequence for prediction"""
        X = df[self.feature_cols].values[-seq_len:]
        return torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    
    def test_week(self, ticker, start_date=None, num_days=5):
        """Test model predictions over a week"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=14)
        
        print(f"\n{'='*60}")
        print(f"Live Testing: {ticker} for {num_days} days")
        print(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}\n")
        
        results = {
            'dates': [], 'predictions': [], 'actual_returns': [],
            'signals': [], 'pnl': [], 'confidence': []
        }
        
        self.cash = 10000
        self.shares = 0
        portfolio_values = [10000]
        
        current_date = start_date
        days_processed = 0
        
        while days_processed < num_days:
            try:
                # Fetch data
                df = self.fetch_recent_data(ticker, current_date, days=100)
                if len(df) < 30:
                    print(f"  Skipping {current_date.strftime('%Y-%m-%d')} - insufficient data")
                    current_date += timedelta(days=1)
                    continue
                
                # Prepare features
                df = self.prepare_features(df)
                X = self.create_sequence(df, seq_len=30).to(self.device)
                
                # Predict
                self.model.eval()
                with torch.no_grad():
                    pred_return, action_logits, conf = self.model(X)
                    signal = torch.argmax(action_logits, dim=1).item() - 1
                    
                    # Filter by confidence
                    if conf.item() < 0.4:
                        signal = 0
                
                # Execute trade
                current_price = df['Close'].iloc[-1]
                
                if signal == 1 and self.shares == 0:
                    shares_to_buy = (self.cash * 0.3) / current_price
                    self.shares = shares_to_buy
                    self.cash -= shares_to_buy * current_price
                    
                elif signal == -1 and self.shares > 0:
                    proceeds = self.shares * current_price
                    self.cash += proceeds
                    self.shares = 0
                
                # Get actual return
                next_date = current_date + timedelta(days=1)
                next_df = self.fetch_recent_data(ticker, next_date + timedelta(days=1), days=5)
                
                if len(next_df) > 0:
                    next_price = next_df['Close'].iloc[-1]
                    actual_return = (next_price - current_price) / current_price
                else:
                    actual_return = 0.0
                
                # Calculate P&L
                portfolio_value = self.cash + self.shares * current_price
                daily_pnl = portfolio_value - portfolio_values[-1]
                portfolio_values.append(portfolio_value)
                
                # Store results
                results['dates'].append(current_date.strftime('%Y-%m-%d'))
                results['predictions'].append(pred_return.item())
                results['actual_returns'].append(actual_return)
                results['signals'].append(signal)
                results['pnl'].append(daily_pnl)
                results['confidence'].append(conf.item())
                
                signal_text = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}[signal]
                print(f"Day {days_processed+1} ({current_date.strftime('%Y-%m-%d')}): "
                      f"Pred={pred_return.item()*100:+.2f}% | "
                      f"Actual={actual_return*100:+.2f}% | "
                      f"Signal={signal_text} | "
                      f"Conf={conf.item()*100:.0f}% | "
                      f"P&L=${daily_pnl:+.2f}")
                
                days_processed += 1
                
            except Exception as e:
                print(f"  Error on {current_date.strftime('%Y-%m-%d')}: {e}")
            
            current_date += timedelta(days=1)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        print(f"\n{'='*60}")
        print("LIVE TEST RESULTS")
        print(f"{'='*60}")
        for key, value in metrics.items():
            print(f"{key:.<40} {value:>10.2f}")
        print(f"{'='*60}\n")
        
        return {'results': results, 'metrics': metrics}
    
    def _calculate_metrics(self, results):
        """Calculate performance metrics"""
        if not results['pnl']:
            return {
                'Total Return (%)': 0.0,
                'Win Rate (%)': 0.0,
                'Accuracy (%)': 0.0,
                'Avg P&L ($)': 0.0
            }
        
        total_return = (sum(results['pnl']) / 10000) * 100
        
        wins = [p for p in results['pnl'] if p > 0]
        win_rate = len(wins) / len(results['pnl']) * 100
        
        correct = sum(
            1 for pred, actual in zip(results['predictions'], results['actual_returns'])
            if (pred > 0 and actual > 0) or (pred < 0 and actual < 0)
        )
        accuracy = (correct / len(results['predictions']) * 100)
        
        return {
            'Total Return (%)': total_return,
            'Win Rate (%)': win_rate,
            'Accuracy (%)': accuracy,
            'Avg P&L ($)': np.mean(results['pnl'])
        }


# ==============================================================================
# 🧠 TRAINING FUNCTION
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
            
            # Check for NaN
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
# 📊 SIGNAL GENERATION FUNCTION
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
# ⚙️ MAIN PIPELINE FUNCTION
# ==============================================================================

def improved_pipeline_with_ensemble(
    ticker='AAPL',
    period='5y',
    seq_len=30,
    batch_size=64,
    epochs=150,
    use_ensemble=True,
    test_live=True,
    enable_shorting=True,
    export_results=True
):
    """
    Complete enhanced trading pipeline
    """
    print(f"\n{'='*60}")
    print(f"Running Enhanced Pipeline for {ticker}")
    print(f"Ensemble: {use_ensemble} | Live Testing: {test_live}")
    print(f"{'='*60}\n")
    
    # 1. Download data
    print("Step 1: Downloading price data...")
    df = download_prices(ticker, period=period)
    print(f"  Downloaded {len(df)} days of data")
    
    # 2. Create technical features
    print("\nStep 2: Creating technical features...")
    df = make_advanced_features(df)
    
    # 3. Add news features
    print("\nStep 3: Adding news sentiment...")
    # FIX: Removed code that introduced look-ahead bias by statically assigning news features.
    print("  Skipping static news feature integration to prevent look-ahead bias.")
    
    # 4. Create labels
    print("\nStep 4: Creating labels...")
    df = create_labels(df, threshold_buy=0.01, threshold_sell=-0.01)
    
    # 5. Build sequences
    print("\nStep 5: Building sequences...")
    feature_cols = [
        'ret1', 'ret5', 'ret20', 'log_ret1', 'daily_range',
        'close_to_ma20', 'rsi14', 'macd', 'volatility20',
        'volume_ratio' # Removed 'news_score' and 'news_momentum'
    ]
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    X, y_ret, y_act, indices, final_features = build_sequences_with_labels(
        df, seq_len=seq_len, feature_cols=feature_cols
    )
    
    print(f"[build_sequences] Created {len(X)} sequences")
    print(f"  - Features: {X.shape[2]}")
    action_counts = pd.Series(y_act).value_counts().sort_index()
    print(f"  - Action distribution: BUY={action_counts.get(2, 0)}, HOLD={action_counts.get(1, 0)}, SELL={action_counts.get(0, 0)}")
    
    # 6. Scale
    print("\nStep 6: Scaling features...")
    scaler = RobustScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(X.shape)
    
    # 7. Split data
    print("\nStep 7: Splitting data...")
    n_test = int(len(X) * 0.15)
    n_val = int(len(X) * 0.15)
    n_train = len(X) - n_val - n_test
    
    X_train, y_ret_train, y_act_train = X[:n_train], y_ret[:n_train], y_act[:n_train]
    X_val, y_ret_val, y_act_val = X[n_train:n_train+n_val], y_ret[n_train:n_train+n_val], y_act[n_train:n_train+n_val]
    X_test, y_ret_test, y_act_test = X[n_train+n_val:], y_ret[n_train+n_val:], y_act[n_train+n_val:]
    idx_test = indices[n_train+n_val:]
    
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Convert to PyTorch
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 8. Create and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStep 8: Creating model (device: {device})...")
    
    if use_ensemble:
        print("  Using Stacking Ensemble (LSTM + GRU + Transformer)")
        model = StackingEnsemble(input_size=X.shape[2], hidden_size=64).to(device)
    else:
        print("  Using Advanced LSTM")
        model = AdvancedLSTM(input_size=X.shape[2], hidden_size=64, num_layers=2).to(device)
    
    loss_fn = CombinedLoss(alpha=1.0, beta=0.3, gamma=0.1)
    
    model, history = train_model_with_validation(
        model, train_loader, val_loader, loss_fn, device,
        epochs=epochs, patience=15, lr=5e-4
    )
    
    # 9. Generate predictions on test set
    print("\nStep 9: Generating predictions on test set...")
    model.eval()
    
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred_ret, action_logits, confidence = model(X_test_t)
        action_probs = torch.softmax(action_logits, dim=1)
    
    signals = generate_signals(pred_ret, action_probs, confidence,
                               return_threshold=0.003, conf_threshold=0.4)
    
    signals_series = pd.Series(signals, index=idx_test)
    confidence_series = pd.Series(confidence.cpu().numpy(), index=idx_test)
    
    print(f"\n  Signal distribution:")
    print(f"    BUY:  {np.sum(signals == 1)}")
    print(f"    HOLD: {np.sum(signals == 0)}")
    print(f"    SELL: {np.sum(signals == -1)}")
    
    # 10. Backtest
    print("\nStep 10: Running backtest...")
    price_for_bt = df.loc[idx_test][['Open', 'Close', 'High', 'Low']]
    
    backtester = AdvancedBacktest(
        init_cash=10000,
        max_position_pct=0.3,
        transaction_cost=0.001,
        stop_loss_pct=0.05,
        enable_shorting=enable_shorting
    )
    
    bt_results = backtester.run(price_for_bt, signals_series, confidence_series)
    
    # 11. Calculate metrics
    print("\nStep 11: Performance Metrics")
    print("="*60)
    metrics = backtester.calculate_metrics(bt_results)
    
    for metric, value in metrics.items():
        # Check for both standard float and NumPy float for printing format
        if isinstance(value, float) or isinstance(value, np.floating):
            print(f"  {metric:.<30} {value:>10.2f}")
        else:
            print(f"  {metric:.<30} {value:>10}")
    
    # 12. Live Testing
    live_results = None
    if test_live:
        print("\nStep 12: Live Testing...")
        tester = LiveTester(model, scaler, final_features)
        live_results = tester.test_week(ticker, num_days=5)
    
    # 13. Export results
    if export_results:
        print("\nStep 13: Exporting results...")
        
        # --- APPLYING THE JSON FIX HERE ---
        # Ensure all NumPy types are converted to standard Python types
        standard_metrics = convert_numpy_types(metrics)
        
        export_data = {
            'ticker': ticker,
            'period': period,
            'use_ensemble': use_ensemble,
            'metrics': standard_metrics,  # Use cleaned metrics
            'backtest': bt_results.reset_index().to_dict('records'),
            'signals': convert_numpy_types(signals_series.to_dict()),
            'confidence': convert_numpy_types(confidence_series.to_dict()),
            'live_testing': convert_numpy_types(live_results) if live_results else None
        }

        # Convert dates to strings in the backtest data for consistency
        for record in export_data['backtest']:
            for key, value in record.items():
                if isinstance(value, (pd.Timestamp, datetime)):
                    record[key] = value.strftime('%Y-%m-%d')
                else:
                    # Safely convert any lingering NumPy or non-serializable types
                    record[key] = convert_numpy_types(value)

        # === Export configuration ===
        export_folder = "data/results"  # Change to your desired folder (e.g. "./output" or "/data")
        os.makedirs(export_folder, exist_ok=True)  # Create folder if it doesn't exist
        output_file = os.path.join(export_folder, f"model_results_{ticker}.json")

        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Results exported to {output_file}")
        print("Upload this file to the dashboard to visualize!")
    
    # 14. Plot results
    print("\nStep 14: Generating plots...")
    backtester.plot_results(bt_results, ticker=ticker)
    
    # Plot training history
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(history['train_loss'], label='Train Loss')
    # ax.plot(history['val_loss'], label='Validation Loss')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Loss')
    # ax.set_title('Training History')
    # ax.legend()
    # ax.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    
    return {
        'model': model,
        'backtest': bt_results,
        'metrics': metrics,
        'signals': signals_series,
        'confidence': confidence_series,
        'scaler': scaler,
        'feature_cols': final_features,
        'live_results': live_results,
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
        test_live=True, 
        enable_shorting=True,
        export_results=True
    )
    