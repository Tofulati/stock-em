import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

from src.config import SEQ_LEN
from src.data_ingest import download_prices
from src.features import make_advanced_features, create_labels, build_sequences_with_labels
from src.models.lstm_model import AdvancedLSTM, CombinedLoss
from src.backtest import AdvancedBacktest

# News sentiment using FinBERT
def init_news_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, device

def score_news_batch(headlines, tokenizer, model, device, batch_size=32):
    """Score news headlines in batches for efficiency"""
    all_scores = []
    
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, 
                          padding=True, max_length=512).to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # positive - negative scores
            scores = probs[:, 0] - probs[:, 1]
            all_scores.extend(scores)
    
    return np.array(all_scores)


def get_fundamentals(ticker):
    """Get fundamental data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        pe = info.get('trailingPE', np.nan)
        pb = info.get('priceToBook', np.nan)
        div = info.get('dividendYield', np.nan)
        ps = info.get('priceToSalesTrailing12Months', np.nan)
        return pe, pb, div, ps
    except Exception as e:
        print(f"Warning: Could not fetch fundamentals: {e}")
        return np.nan, np.nan, np.nan, np.nan


def train_model_with_validation(model, train_loader, val_loader, loss_fn, device, 
                                epochs=100, patience=15, lr=1e-3):
    """Train model with early stopping"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = np.inf
    best_model_state = model.state_dict()  # Initialize with current state
    counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
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
            
            # Check for NaN in predictions
            if torch.isnan(pred_ret).any() or torch.isnan(action_logits).any():
                print(f"Warning: NaN detected in predictions at epoch {epoch}")
                print(f"  pred_ret has NaN: {torch.isnan(pred_ret).any()}")
                print(f"  action_logits has NaN: {torch.isnan(action_logits).any()}")
                print(f"  Input stats - min: {xb.min():.4f}, max: {xb.max():.4f}, mean: {xb.mean():.4f}")
                # Skip this batch
                continue
            
            loss, loss_dict = loss_fn(pred_ret, action_logits, confidence, y_ret, y_act)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch}, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping
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
                
                # Skip if NaN
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


def generate_signals(pred_returns, action_probs, confidence, 
                     return_threshold=0.005, conf_threshold=0.5):
    """
    Convert model predictions to trading signals
    
    Returns:
        signals: -1 (sell/short), 0 (hold), 1 (buy/long)
    """
    # Get action predictions (0=sell, 1=hold, 2=buy)
    actions = torch.argmax(action_probs, dim=1).cpu().numpy()
    
    # Convert to trading signals
    signals = actions - 1  # Convert [0,1,2] to [-1,0,1]
    
    # Filter by confidence
    low_conf_mask = confidence.cpu().numpy() < conf_threshold
    signals[low_conf_mask] = 0  # Hold when confidence is low
    
    # Additional filter: if predicted return is too small, hold
    pred_ret_np = pred_returns.cpu().numpy()
    small_ret_mask = np.abs(pred_ret_np) < return_threshold
    signals[small_ret_mask] = 0
    
    return signals


def improved_pipeline(
    ticker='AAPL',
    period='5y',
    seq_len=30,
    batch_size=64,
    epochs=150,
    val_split=0.15,
    test_split=0.15,
    enable_news=False,
    enable_shorting=True
):
    """
    Complete improved trading pipeline with:
    - Advanced features
    - Multi-task LSTM
    - Proper train/val/test split
    - Long/short backtesting
    """
    
    print(f"\n{'='*60}")
    print(f"Running Improved Pipeline for {ticker}")
    print(f"{'='*60}\n")
    
    # ========== 1. Download and prepare data ==========
    print("Step 1: Downloading price data...")
    df = download_prices(ticker, period=period)
    print(f"  Downloaded {len(df)} days of data")
    
    # ========== 2. Create advanced features ==========
    print("\nStep 2: Creating advanced technical features...")
    df = make_advanced_features(df)
    
    # ========== 3. Add fundamentals ==========
    print("\nStep 3: Adding fundamental data...")
    pe, pb, div, ps = get_fundamentals(ticker)
    df['PE'] = pe
    df['PB'] = pb
    df['DivYield'] = div
    df['PS'] = ps
    df[['PE', 'PB', 'DivYield', 'PS']] = df[['PE', 'PB', 'DivYield', 'PS']].bfill().ffill()
    
    # ========== 4. Add news sentiment (optional) ==========
    if enable_news:
        print("\nStep 4: Scoring news sentiment...")
        tokenizer, news_model, news_device = init_news_model()
        # Placeholder: replace with actual news headlines
        headlines = [f"Stock market update for {ticker}"] * len(df)
        df['news_score'] = score_news_batch(headlines, tokenizer, news_model, news_device)
    else:
        df['news_score'] = 0.0
    
    # ========== 5. Create labels ==========
    print("\nStep 5: Creating target labels...")
    df = create_labels(df, threshold_buy=0.01, threshold_sell=-0.01)
    
    # ========== 6. Select and scale features ==========
    print("\nStep 6: Selecting and scaling features...")
    feature_cols = [
        # Returns
        'ret1', 'ret5', 'ret20', 'log_ret1',
        # Price patterns
        'daily_range', 'open_close_spread',
        # Moving averages
        'close_to_ma5', 'close_to_ma20', 'close_to_ma50', 'close_to_ma200',
        'pct_from_52w_high', 'pct_from_52w_low',
        # Momentum
        'rsi14', 'rsi7', 'macd', 'macd_hist', 'roc10', 'roc20',
        # Volatility
        'volatility5', 'volatility20', 'bb_position', 'bb_width', 'atr_pct',
        # Volume
        'volume_ratio', 'volume_change', 'obv_ratio', 'close_to_vwap',
        # Statistics
        'ret_skew20', 'ret_kurt20', 'ret_autocorr5',
        # Fundamentals
        'PE', 'PB', 'DivYield', 'PS',
        # News
        'news_score'
    ]
    
    # Filter features that exist
    feature_cols = [f for f in feature_cols if f in df.columns]
    print(f"  Using {len(feature_cols)} features")
    
    # === DATA QUALITY CHECKS ===
    print("\n  Data quality checks:")
    
    # Check for infinite values
    inf_counts = {}
    for col in feature_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        print(f"  ⚠️  Found infinite values in {len(inf_counts)} features:")
        for col, count in list(inf_counts.items())[:5]:
            print(f"      {col}: {count} inf values")
    
    # Check for NaN values before scaling
    nan_counts = {}
    for col in feature_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_counts[col] = nan_count
    
    if nan_counts:
        print(f"  ⚠️  Found NaN values in {len(nan_counts)} features:")
        for col, count in list(nan_counts.items())[:5]:
            print(f"      {col}: {count} NaN values")
    
    # Replace infinite values with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values using forward fill, then backward fill, then zero
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)
    
    # Check if any NaN remain
    remaining_nan = df[feature_cols].isna().sum().sum()
    if remaining_nan > 0:
        print(f"  ⚠️  WARNING: {remaining_nan} NaN values remain after filling!")
        df[feature_cols] = df[feature_cols].fillna(0)
    else:
        print(f"  ✓ No NaN values after cleaning")
    
    # Check for extreme values before scaling
    print("\n  Feature value ranges (before scaling):")
    for col in feature_cols[:5]:  # Show first 5
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        print(f"      {col:20s}: [{min_val:>10.4f}, {max_val:>10.4f}] mean={mean_val:>10.4f}")
    
    # Robust scaling (better for financial data with outliers)
    scaler = RobustScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Final sanity check after scaling
    print("\n  After scaling:")
    has_nan = df[feature_cols].isna().any().any()
    has_inf = np.isinf(df[feature_cols].values).any()
    print(f"    Has NaN: {has_nan}")
    print(f"    Has Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("  ⚠️  ERROR: Still have NaN/Inf after scaling! Clipping to [-10, 10]")
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        df[feature_cols] = df[feature_cols].clip(-10, 10)
    else:
        print("  ✓ Data is clean and ready for training")
    
    # ========== 7. Build sequences ==========
    print("\nStep 7: Building sequences...")
    X, y_ret, y_act, indices, final_features = build_sequences_with_labels(
        df, seq_len=seq_len, feature_cols=feature_cols
    )
    
    # ========== 8. Train/Val/Test split (time-based) ==========
    print("\nStep 8: Splitting data (time-series aware)...")
    n_samples = len(X)
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val - n_test
    
    X_train = X[:n_train]
    y_ret_train = y_ret[:n_train]
    y_act_train = y_act[:n_train]
    
    X_val = X[n_train:n_train+n_val]
    y_ret_val = y_ret[n_train:n_train+n_val]
    y_act_val = y_act[n_train:n_train+n_val]
    
    X_test = X[n_train+n_val:]
    y_ret_test = y_ret[n_train+n_val:]
    y_act_test = y_act[n_train+n_val:]
    idx_test = indices[n_train+n_val:]
    
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Convert to PyTorch tensors
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
    
    # ========== 9. Train model ==========
    print("\nStep 9: Training advanced LSTM model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Use smaller model to reduce NaN risk
    model = AdvancedLSTM(
        input_size=X.shape[2],
        hidden_size=64,  # Reduced from 128
        num_layers=2,    # Reduced from 3
        num_heads=2,     # Reduced from 4
        dropout=0.2      # Reduced from 0.3
    ).to(device)
    
    # Reduce loss weights to prevent instability
    loss_fn = CombinedLoss(alpha=1.0, beta=0.3, gamma=0.1)
    
    model, history = train_model_with_validation(
        model, train_loader, val_loader, loss_fn, device,
        epochs=epochs, patience=15, lr=5e-4  # Lower learning rate
    )
    
    # ========== 10. Generate predictions on test set ==========
    print("\nStep 10: Generating predictions on test set...")
    model.eval()
    
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred_ret, action_logits, confidence = model(X_test_t)
        action_probs = torch.softmax(action_logits, dim=1)
    
    # Generate trading signals
    signals = generate_signals(pred_ret, action_probs, confidence,
                               return_threshold=0.003, conf_threshold=0.4)
    
    signals_series = pd.Series(signals, index=idx_test)
    confidence_series = pd.Series(confidence.cpu().numpy(), index=idx_test)
    
    print(f"\n  Signal distribution:")
    print(f"    BUY:  {np.sum(signals == 1)}")
    print(f"    HOLD: {np.sum(signals == 0)}")
    print(f"    SELL: {np.sum(signals == -1)}")
    
    # ========== 11. Backtest ==========
    print("\nStep 11: Running backtest...")
    price_for_bt = df.loc[idx_test][['Open', 'Close', 'High', 'Low']]
    
    backtester = AdvancedBacktest(
        init_cash=10000,
        max_position_pct=0.3,
        transaction_cost=0.001,
        stop_loss_pct=0.05,
        enable_shorting=enable_shorting
    )
    
    bt_results = backtester.run(price_for_bt, signals_series, confidence_series)
    
    # ========== 12. Calculate and display metrics ==========
    print("\nStep 12: Performance Metrics")
    print("="*60)
    metrics = backtester.calculate_metrics(bt_results)
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric:.<30} {value:>10.2f}")
        else:
            print(f"  {metric:.<30} {value:>10}")
    
    # ========== 13. Plot results ==========
    print("\nStep 13: Generating plots...")
    backtester.plot_results(bt_results, ticker=ticker)
    
    # Plot training history
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
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
        'feature_cols': final_features
    }


if __name__ == '__main__':
    # Run pipeline
    results = improved_pipeline(
        ticker='INTC',
        period='5y',
        seq_len=30,
        batch_size=64,
        epochs=150,
        enable_news=False,  # Set to True if you have real news data
        enable_shorting=True
    )