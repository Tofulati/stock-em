import numpy as np
import pandas as pd
import torch
from src.config import TICKERS, SEQ_LEN
from src.data_ingest import download_prices, load_prices
from src.features import make_technical_features, build_sequences
from src.models.lstm_model import LSTMRegressor
from src.models.news_model import score_text
from src.ensemble import combine_scores
from src.backtest import simple_backtest

def toy_pipeline(ticker='AAPL'):
    print('Downloading price')
    df = download_prices(ticker, period='1y')
    df = make_technical_features(df)
    print(df.head())
    print("Data shape:", df.shape)
    print("SEQ_LEN =", SEQ_LEN)

    # Build sequences for LSTM
    feature_cols = ['ret1','ma5','ma20','pct_from_52w_high','Volume']
    X, idx = build_sequences(df, features=feature_cols)

    # Labels: next-day return
    y = df['ret1'].values[SEQ_LEN:SEQ_LEN+len(X)]

    # Quick LSTM training
    device = torch.device('cpu')
    model = LSTMRegressor(input_size=X.shape[2], hidden_size=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    for epoch in range(10):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()
        if epoch % 5 == 0:
            print('epoch', epoch, 'loss', loss.item())

    model.eval()
    with torch.no_grad():
        rnn_preds = model(X_t).numpy()

    # Attach predictions back to index
    preds_series = pd.Series(rnn_preds, index=idx)

    # Toy news scoring
    news_scores = pd.Series(0.0, index=idx)

    # Simple PE z-score
    pe_z = pd.Series(np.random.randn(len(idx)) * 0.1, index=idx)
    pct_from_high = df.loc[idx]['pct_from_52w_high']

    final_score = combine_scores(preds_series, news_scores, pe_z, pct_from_high)

    # Signals in [-1,1]
    signals = (final_score - final_score.mean()) / (final_score.std() + 1e-8)
    signals = signals.clip(-1,1)

    # Backtest: using preserved 'Open','Close'
    price_for_bt = df.loc[idx][['Open','Close']]
    bt = simple_backtest(price_for_bt, signals)
    print(bt.tail())

if __name__ == '__main__':
    toy_pipeline('AAPL')
