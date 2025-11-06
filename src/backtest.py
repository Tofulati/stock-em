import pandas as pd
import numpy as np

def simple_backtest(df_price, signals, init_cash=10000, position_pct=0.1):
    # df_price: DataFrame indexed by date with 'Open','Close'
    # signals: Series indexed by date with values in [-1,1] (positive -> long)
    cash = init_cash
    shares = 0.0
    history = []
    for date in signals.index:
        sig = signals.loc[date]
    # execute at next day's Open (if available)
        try:
            price = df_price.loc[date]['Open']
        except KeyError:
            continue
        target_value = cash * position_pct * sig
        target_shares = target_value / price
        # simple: buy/sell delta
        delta = target_shares - shares
        cost = delta * price
        cash -= cost
        shares += delta
        nav = cash + shares * df_price.loc[date]['Close']
        history.append({'date': date, 'cash': cash, 'shares': shares, 'nav': nav})
    return pd.DataFrame(history).set_index('date')