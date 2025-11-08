import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew, kurtosis

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    rsi = 100 - 100 / (1 + ma_up / (ma_down + 1e-10))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    bb_position = (series - lower_band) / (upper_band - lower_band + 1e-10)
    return bb_position, upper_band, lower_band

def compute_atr(high, low, close, window=14):
    """Average True Range - volatility indicator"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window).mean()
    return atr

def compute_obv(close, volume):
    """On-Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def compute_vwap(high, low, close, volume):
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def make_advanced_features(df):
    """
    Create comprehensive technical features for stock prediction
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # ============ Price-based features ============
    # Returns at multiple horizons
    df['ret1'] = df['close'].pct_change()
    df['ret5'] = df['close'].pct_change(5)
    df['ret20'] = df['close'].pct_change(20)
    
    # Log returns (more stable for modeling)
    df['log_ret1'] = np.log(df['close'] / df['close'].shift(1))
    
    # Intraday range
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['open_close_spread'] = (df['close'] - df['open']) / df['open']
    
    # Moving averages - multiple timeframes
    for window in [5, 10, 20, 50, 200]:
        df[f'ma{window}'] = df['close'].rolling(window).mean()
        df[f'close_to_ma{window}'] = df['close'] / df[f'ma{window}'] - 1
    
    # Exponential moving averages
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Distance from 52-week high/low
    window_252 = min(252, len(df)//2)
    df['pct_from_52w_high'] = df['close'] / df['close'].rolling(window_252).max() - 1
    df['pct_from_52w_low'] = df['close'] / df['close'].rolling(window_252).min() - 1
    
    # ============ Momentum Indicators ============
    df['rsi14'] = compute_rsi(df['close'], 14)
    df['rsi7'] = compute_rsi(df['close'], 7)
    
    macd, macd_signal, macd_hist = compute_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    
    # Rate of change
    df['roc10'] = df['close'].pct_change(10)
    df['roc20'] = df['close'].pct_change(20)
    
    # ============ Volatility Indicators ============
    df['volatility5'] = df['ret1'].rolling(5).std()
    df['volatility20'] = df['ret1'].rolling(20).std()
    df['volatility60'] = df['ret1'].rolling(60).std()
    
    # Bollinger Bands
    df['bb_position'], df['bb_upper'], df['bb_lower'] = compute_bollinger_bands(df['close'], 20)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    
    # ATR - Average True Range
    df['atr14'] = compute_atr(df['high'], df['low'], df['close'], 14)
    df['atr_pct'] = df['atr14'] / df['close']
    
    # ============ Volume Indicators ============
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma20'] + 1)
    df['volume_change'] = df['volume'].pct_change()
    
    # On-Balance Volume
    df['obv'] = compute_obv(df['close'], df['volume'])
    df['obv_ma20'] = df['obv'].rolling(20).mean()
    df['obv_ratio'] = df['obv'] / (df['obv_ma20'] + 1)
    
    # VWAP
    df['vwap'] = compute_vwap(df['high'], df['low'], df['close'], df['volume'])
    df['close_to_vwap'] = df['close'] / df['vwap'] - 1
    
    # ============ Statistical Features ============
    # Rolling statistics of returns
    df['ret_skew20'] = df['ret1'].rolling(20).apply(lambda x: skew(x, nan_policy='omit'), raw=False)
    df['ret_kurt20'] = df['ret1'].rolling(20).apply(lambda x: kurtosis(x, nan_policy='omit'), raw=False)
    
    # Autocorrelation (momentum persistence)
    df['ret_autocorr5'] = df['ret1'].rolling(20).apply(lambda x: pd.Series(x).autocorr(5), raw=False)
    
   # ============ Time-based Features ============
    try:
        if 'date' in df.columns:
            date_col = pd.to_datetime(df['date'])
        elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
            date_col = pd.to_datetime(df.index)
        else:
            date_col = None
        
        if date_col is not None:
            # Use .dt accessor for pandas datetime properties
            df['day_of_week'] = date_col.dt.dayofweek if hasattr(date_col, 'dt') else date_col.dayofweek
            df['day_of_month'] = date_col.dt.day if hasattr(date_col, 'dt') else date_col.day
            df['month'] = date_col.dt.month if hasattr(date_col, 'dt') else date_col.month
            df['quarter'] = date_col.dt.quarter if hasattr(date_col, 'dt') else date_col.quarter
            
            # Cyclical encoding for time features
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    except Exception as e:
        print(f"Warning: Could not create time-based features: {e}")
    
    # ============ Clean up ============
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.bfill().ffill()
    
    # Restore capitalized columns for compatibility
    df.rename(columns={
        'close': 'Close',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'volume': 'Volume'
    }, inplace=True)
    
    print(f"[make_advanced_features] Created {len(df.columns)} features, {len(df)} rows")
    
    return df


def create_labels(df, threshold_buy=0.01, threshold_sell=-0.01):
    """
    Create action labels: 
    - 2 = BUY (strong upward movement expected)
    - 1 = HOLD (neutral)
    - 0 = SELL/SHORT (strong downward movement expected)
    """
    df = df.copy()
    
    # Future return (what we're trying to predict)
    df['future_ret'] = df['Close'].shift(-1) / df['Close'] - 1
    
    # Create action labels based on thresholds
    df['action'] = 1  # default: HOLD
    df.loc[df['future_ret'] > threshold_buy, 'action'] = 2  # BUY
    df.loc[df['future_ret'] < threshold_sell, 'action'] = 0  # SELL
    
    return df


def build_sequences_with_labels(df, seq_len=20, feature_cols=None):
    """
    Build sequences for training with return and action labels
    """
    if 'future_ret' not in df.columns or 'action' not in df.columns:
        raise ValueError("Must call create_labels() before build_sequences_with_labels()")
    
    if feature_cols is None:
        # Exclude target variables and non-numeric columns
        exclude = ['future_ret', 'action', 'Close', 'Open', 'High', 'Low', 'Volume', 
                   'date', 'Date', 'bb_upper', 'bb_lower', 'obv_ma20', 'volume_ma20']
        feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64]]
    
    # Check all features exist
    for f in feature_cols:
        if f not in df.columns:
            raise ValueError(f"Feature '{f}' not found in DataFrame")
    
    X_data = df[feature_cols].values
    y_return = df['future_ret'].values
    y_action = df['action'].values
    
    X, y_ret, y_act, indices = [], [], [], []
    
    for i in range(len(X_data) - seq_len):
        # Skip if future return is NaN (last row)
        if np.isnan(y_return[i + seq_len]):
            continue
            
        X.append(X_data[i:i+seq_len])
        y_ret.append(y_return[i + seq_len])
        y_act.append(y_action[i + seq_len])
        indices.append(df.index[i + seq_len])
    
    X = np.stack(X).astype(np.float32)
    y_ret = np.array(y_ret, dtype=np.float32)
    y_act = np.array(y_act, dtype=np.int64)
    
    print(f"[build_sequences] Created {len(X)} sequences")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Action distribution: BUY={np.sum(y_act==2)}, HOLD={np.sum(y_act==1)}, SELL={np.sum(y_act==0)}")
    
    return X, y_ret, y_act, indices, feature_cols