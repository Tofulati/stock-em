import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AdvancedBacktest:
    def __init__(
        self, 
        init_cash=10000,
        max_position_pct=0.3,
        transaction_cost=0.001,
        stop_loss_pct=0.05,
        enable_shorting=True
    ):
        self.init_cash = init_cash
        self.max_position_pct = max_position_pct
        self.transaction_cost = transaction_cost
        self.stop_loss_pct = stop_loss_pct
        self.enable_shorting = enable_shorting
        
    def run(self, df_price, signals, confidence=None):
        df_price = df_price.copy()

        # Align signals to price data (drop missing)
        signals = signals.reindex(df_price.index).fillna(0)
        if confidence is not None:
            confidence = confidence.reindex(df_price.index).fillna(1.0)

        cash = self.init_cash
        shares = 0.0
        avg_entry = None

        history = []
        
        for date, sig in signals.items():
            row = df_price.loc[date]
            open_p = row["Open"]
            close_p = row["Close"]
            high_p = row.get("High", close_p)
            low_p  = row.get("Low", close_p)

            conf = confidence.loc[date] if confidence is not None else 1.0

            # ----- STOP LOSS -----
            if shares != 0 and avg_entry is not None:
                if shares > 0:  # long stop
                    if low_p <= avg_entry * (1 - self.stop_loss_pct):
                        stop = avg_entry * (1 - self.stop_loss_pct)
                        cash += shares * stop * (1 - self.transaction_cost)
                        shares = 0
                        avg_entry = None

                elif shares < 0:  # short stop
                    if high_p >= avg_entry * (1 + self.stop_loss_pct):
                        stop = avg_entry * (1 + self.stop_loss_pct)
                        cash -= abs(shares) * stop * (1 + self.transaction_cost)
                        shares = 0
                        avg_entry = None
            
            # ----- SIGNAL ACTIONS -----
            if sig > 0:
                # close shorts
                if shares < 0:
                    cash -= abs(shares) * open_p * (1 + self.transaction_cost)
                    shares = 0
                    avg_entry = None

                if shares == 0 and cash > 0:
                    pos_val = cash * self.max_position_pct * conf
                    new_shares = pos_val / open_p
                    cost = new_shares * open_p * (1 + self.transaction_cost)
                    if cost <= cash:
                        cash -= cost
                        shares = new_shares
                        avg_entry = open_p

            elif sig < 0:
                if not self.enable_shorting:
                    if shares > 0:
                        cash += shares * open_p * (1 - self.transaction_cost)
                        shares = 0
                        avg_entry = None
                else:
                    # close longs
                    if shares > 0:
                        cash += shares * open_p * (1 - self.transaction_cost)
                        shares = 0
                        avg_entry = None

                    if shares == 0 and cash > 0:
                        pos_val = cash * self.max_position_pct * conf
                        new_shares = pos_val / open_p
                        cash += new_shares * open_p * (1 - self.transaction_cost)
                        shares = -new_shares
                        avg_entry = open_p

            else:
                if shares > 0:
                    cash += shares * open_p * (1 - self.transaction_cost)
                    shares = 0
                    avg_entry = None
                elif shares < 0:
                    cash -= abs(shares) * open_p * (1 + self.transaction_cost)
                    shares = 0
                    avg_entry = None

            # ----- NAV CALCULATION (correct unified formula) -----
            position_value = shares * close_p       # works for long (+) and short (-)
            nav = cash + position_value

            history.append({
                "date": date,
                "cash": cash,
                "shares": shares,
                "position_value": position_value,
                "nav": nav,
                "signal": sig,
                "confidence": conf
            })

        df_result = pd.DataFrame(history).set_index("date")
        return df_result

    def calculate_metrics(self, df):
        nav = df["nav"]

        returns = nav.pct_change().dropna()
        total_return = (nav.iloc[-1]/nav.iloc[0] - 1) * 100

        years = len(nav)/252
        annualized_return = ((nav.iloc[-1]/nav.iloc[0])**(1/years)-1)*100 if years > 0 else 0

        sharpe = (returns.mean()/returns.std() * np.sqrt(252)) if returns.std() != 0 else 0

        cumulative = (1+returns).cumprod()
        max_dd = (cumulative / cumulative.cummax() - 1).min() * 100

        # Count real trades
        trade_changes = df["shares"].apply(lambda x: 1 if x != 0 else 0)
        trades = (trade_changes.diff() != 0).sum()

        return {
            "Total Return (%)": total_return,
            "Annualized Return (%)": annualized_return,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd,
            "Number of Trades": int(trades),
            "Final NAV": float(nav.iloc[-1])
        }
    
    def plot_results(self, df_backtest, ticker='Stock', save_path=None):
        """Plot backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # NAV over time
        axes[0].plot(df_backtest.index, df_backtest['nav'], label='Portfolio NAV', linewidth=2)
        axes[0].axhline(y=self.init_cash, color='gray', linestyle='--', label='Initial Capital')
        axes[0].set_title(f'{ticker} - Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('NAV ($)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Position sizes
        axes[1].fill_between(df_backtest.index, 0, df_backtest['shares'], 
                             where=(df_backtest['shares'] > 0), color='green', alpha=0.3, label='Long')
        axes[1].fill_between(df_backtest.index, 0, df_backtest['shares'], 
                             where=(df_backtest['shares'] < 0), color='red', alpha=0.3, label='Short')
        axes[1].set_title('Position Sizes', fontsize=12)
        axes[1].set_ylabel('Shares')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Signals
        signal_colors = {-1: 'red', 0: 'gray', 1: 'green'}
        for sig in [-1, 0, 1]:
            mask = df_backtest['signal'] == sig
            axes[2].scatter(df_backtest.index[mask], df_backtest['signal'][mask], 
                          c=signal_colors[sig], alpha=0.6, s=10)
        axes[2].set_title('Trading Signals', fontsize=12)
        axes[2].set_ylabel('Signal')
        axes[2].set_ylim(-1.5, 1.5)
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig