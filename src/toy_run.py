import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification

from src.config import SEQ_LEN
from src.data_ingest import download_prices
from src.features import make_technical_features, build_sequences
from src.models.lstm_model import ImprovedLSTM
from src.ensemble import combine_scores
from src.backtest import simple_backtest

# --- Fundamentals ---
def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    pe = info.get('trailingPE', np.nan)
    pb = info.get('priceToBook', np.nan)
    div = info.get('dividendYield', np.nan)
    return pe, pb, div

# --- News sentiment using FinBERT ---
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model_news = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_news.to(device)
model_news.eval()

def score_news_headlines(headlines):
    scores = []
    for text in headlines:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = model_news(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            score = probs[0] - probs[1]  # positive - negative
            scores.append(score)
    return np.array(scores)

# --- Pipeline ---
def toy_pipeline(ticker='AAPL', seq_len=SEQ_LEN, batch_size=64, epochs=100, val_split=0.1):
    print('Downloading price')
    df = download_prices(ticker, period='10y')
    df = make_technical_features(df)

    # --- Fundamentals ---
    pe, pb, div = get_fundamentals(ticker)
    df['PE'] = pe
    df['PB'] = pb
    df['DivYield'] = div
    df[['PE','PB','DivYield']] = df[['PE','PB','DivYield']].bfill().ffill()

    # --- News headlines placeholder ---
    # Replace with actual daily headlines; here we simulate equal-length text
    headlines = ["Stock news example"] * len(df)
    df['news_score'] = score_news_headlines(headlines)

    # --- Normalize features ---
    feature_cols = [
        'ret1','ma5','ma20','pct_from_52w_high','Volume',
        'PE','PB','DivYield','news_score','rsi14','macd','macd_signal','volatility20'
    ]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # --- Build sequences ---
    X, idx = build_sequences(df, seq_len=seq_len, features=feature_cols)
    y = df['ret1'].values[seq_len:seq_len+len(X)]

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, shuffle=False)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=False)

    # LSTM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = ImprovedLSTM(input_size=X.shape[2], hidden_size=64, num_layers=2).to(device)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Early stopping
    best_val_loss = np.inf
    patience = 10
    counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t.to(device)), y_val_t.to(device)).item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Validation Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_model_state)

    # Full predictions
    model.eval()
    with torch.no_grad():
        rnn_preds = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    preds_series = pd.Series(rnn_preds, index=idx)

    # Combine scores: LSTM + news + PE z-score + pct from high
    pe_z = pd.Series(np.random.randn(len(idx)) * 0.1, index=idx)
    news_scores = df.loc[idx]['news_score']
    pct_from_high = df.loc[idx]['pct_from_52w_high']
    final_score = combine_scores(preds_series, news_scores, pe_z, pct_from_high)

    # Signals [-1,1]
    signals = (final_score - final_score.mean()) / (final_score.std() + 1e-8)
    signals = signals.clip(-1,1)

    # Backtest
    price_for_bt = df.loc[idx][['Open','Close']]
    bt = simple_backtest(price_for_bt, signals)

    # --- Debug prints ---
    print("\n--- Debug info (last 10 rows) ---")
    debug_df = pd.DataFrame({
        'signal': signals,
        'rnn_pred': preds_series,
        'news_score': news_scores,
        'pe_z': pe_z,
        'pct_from_52w_high': pct_from_high,
        'Open': price_for_bt['Open'],
        'Close': price_for_bt['Close'],
        'cash': bt['cash'],
        'shares': bt['shares'],
        'nav': bt['nav']
    })
    print(debug_df.tail(10))

    # Plot NAV
    bt['nav'].plot(title=f"{ticker} Backtest NAV")
    plt.ylabel('NAV ($)')
    plt.show()

    return bt

if __name__ == '__main__':
    toy_pipeline('AAPL')
