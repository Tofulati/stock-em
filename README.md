Stock-em Pipeline
======================


This skeleton repository shows a minimal end-to-end pipeline to:
- ingest historic price & fundamentals (yfinance)
- ingest news (NewsAPI or newspaper3k)
- build features and sequences for an LSTM
- score news with a pretrained finBERT (HuggingFace) model
- combine RNN predictions + news score + fundamentals into a final score
- run a simple backtest (daily bars, simple execution)


Usage (quick):
1. Create a virtualenv and `pip install -r requirements.txt`.
2. Configure `DATA_DIR` and (optionally) NEWS_API_KEY in `src/config.py`.
3. Run `python -m src.toy_run` to run the toy pipeline using yfinance and simulated news.


Notes:
- This is a starting template. Do NOT run live without adding proper error handling and evaluation.
- The code uses yfinance + HuggingFace transformers; these require Internet when first downloaded.

======================
How the Model Works (Step-by-Step)
1. Data Pipeline (data_ingest.py)
- Downloads historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance
- Stores data locally as Parquet files for fast reloading

2. Feature Engineering (features.py)
- Returns: 1-day, 5-day, 20-day returns
- Momentum: RSI (Relative Strength Index), MACD
- Volatility: 20-day rolling standard deviation
- Volume: Volume ratio vs. 20-day average
-  Price position: Distance from 20-day moving average

3. Label Creation (Target Variables)
- Regression Target (y_ret): Actual next-day return percentage
- Classification Target (y_act): Action label
    - BUY (2): If next-day return > +1%
    - HOLD (1): If return between -1% and +1%
    - SELL (0): If return < -1%

4. Sequence Building
The model uses sequences of 30 days (configurable via SEQ_LEN):

5. Model Architecture
- Option A: Advanced LSTM (src/models/lstm_model.py)
- Option B: Stacking Ensemble (ensemble.py)
    - LSTM: Good at long-term dependencies
    - GRU: Faster, captures short-term patterns
    - Transformer: Learns complex relationships

6. Training Process (toy_run.py)

7. Signal Generation