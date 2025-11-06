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
3. Run `python src/toy_run.py` to run the toy pipeline using yfinance and simulated news.


Notes:
- This is a starting template. Do NOT run live without adding proper error handling and evaluation.
- The code uses yfinance + HuggingFace transformers; these require Internet when first downloaded.
