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
        """
        Run backtest with proper signal handling.
        
        Parameters:
        -----------
        df_price : DataFrame with columns ['Open', 'Close', 'High', 'Low']
        signals : Series with values: 1 (buy), -1 (sell), 0 (hold)
                 NOTE: Only trades when signal CHANGES from previous day
        confidence : Series with confidence scores (0-1), optional
        """
        df_price = df_price.copy()

        # Align signals to price data
        signals = signals.reindex(df_price.index).fillna(0)
        if confidence is not None:
            confidence = confidence.reindex(df_price.index).fillna(1.0)

        cash = self.init_cash
        shares = 0.0
        avg_entry = None
        
        # Track previous signal to detect changes
        prev_signal = 0

        history = []
        
        for date, sig in signals.items():
            row = df_price.loc[date]
            open_p = row["Open"]
            close_p = row["Close"]
            high_p = row.get("High", close_p)
            low_p = row.get("Low", close_p)

            conf = confidence.loc[date] if confidence is not None else 1.0
            
            # CRITICAL: Only trade when signal CHANGES
            signal_changed = (sig != prev_signal)

            # ----- STOP LOSS CHECK -----
            stop_triggered = False
            if shares != 0 and avg_entry is not None:
                if shares > 0:  # Long position stop
                    if low_p <= avg_entry * (1 - self.stop_loss_pct):
                        exit_price = avg_entry * (1 - self.stop_loss_pct)
                        cash += shares * exit_price * (1 - self.transaction_cost)
                        shares = 0
                        avg_entry = None
                        stop_triggered = True
                        prev_signal = 0  # Reset signal after stop

                elif shares < 0:  # Short position stop
                    if high_p >= avg_entry * (1 + self.stop_loss_pct):
                        exit_price = avg_entry * (1 + self.stop_loss_pct)
                        cash -= abs(shares) * exit_price * (1 + self.transaction_cost)
                        shares = 0
                        avg_entry = None
                        stop_triggered = True
                        prev_signal = 0  # Reset signal after stop
            
            # ----- SIGNAL ACTIONS (only on signal change) -----
            if not stop_triggered and signal_changed:
                
                if sig > 0:  # NEW BUY signal
                    # Close any short position
                    if shares < 0:
                        cash -= abs(shares) * open_p * (1 + self.transaction_cost)
                        shares = 0
                        avg_entry = None

                    # Enter long if flat
                    if shares == 0 and cash > 0:
                        pos_val = cash * self.max_position_pct * conf
                        new_shares = pos_val / open_p
                        cost = new_shares * open_p * (1 + self.transaction_cost)
                        if cost <= cash:
                            cash -= cost
                            shares = new_shares
                            avg_entry = open_p

                elif sig < 0:  # NEW SELL signal
                    if not self.enable_shorting:
                        # Just exit long positions
                        if shares > 0:
                            cash += shares * open_p * (1 - self.transaction_cost)
                            shares = 0
                            avg_entry = None
                    else:
                        # Close any long position
                        if shares > 0:
                            cash += shares * open_p * (1 - self.transaction_cost)
                            shares = 0
                            avg_entry = None

                        # Enter short if flat
                        if shares == 0 and cash > 0:
                            pos_val = cash * self.max_position_pct * conf
                            new_shares = pos_val / open_p
                            proceeds = new_shares * open_p * (1 - self.transaction_cost)
                            cash += proceeds
                            shares = -new_shares
                            avg_entry = open_p

                elif sig == 0:  # NEW EXIT signal (flatten)
                    if shares > 0:
                        cash += shares * open_p * (1 - self.transaction_cost)
                        shares = 0
                        avg_entry = None
                    elif shares < 0:
                        cash -= abs(shares) * open_p * (1 + self.transaction_cost)
                        shares = 0
                        avg_entry = None
                
                # Update previous signal
                prev_signal = sig

            # ----- NAV CALCULATION -----
            position_value = shares * close_p
            nav = cash + position_value

            history.append({
                "date": date,
                "cash": cash,
                "shares": shares,
                "position_value": position_value,
                "nav": nav,
                "signal": sig,
                "confidence": conf,
                "price": close_p,
                "signal_changed": signal_changed
            })

        df_result = pd.DataFrame(history).set_index("date")
        return df_result

    def calculate_metrics(self, df):
        """Calculate performance metrics"""
        nav = df["nav"]
        returns = nav.pct_change().dropna()
        
        # Basic returns
        total_return = (nav.iloc[-1]/nav.iloc[0] - 1) * 100
        years = len(nav)/252
        annualized_return = ((nav.iloc[-1]/nav.iloc[0])**(1/years)-1)*100 if years > 0 else 0

        # Risk metrics
        sharpe = (returns.mean()/returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        cumulative = (1+returns).cumprod()
        max_dd = (cumulative / cumulative.cummax() - 1).min() * 100

        # Trade counting - only count actual signal changes
        trades = df["signal_changed"].sum()

        # Win rate - track complete trades
        trade_returns = []
        entry_nav = None
        entry_idx = None
        
        for idx, row in df.iterrows():
            # Opening a position
            if row["shares"] != 0 and entry_nav is None:
                entry_nav = row["nav"]
                entry_idx = idx
            # Closing a position
            elif row["shares"] == 0 and entry_nav is not None:
                trade_return = (row["nav"] / entry_nav - 1)
                trade_returns.append(trade_return)
                entry_nav = None
                entry_idx = None
        
        # Handle open position at end
        if entry_nav is not None:
            final_return = (nav.iloc[-1] / entry_nav - 1)
            trade_returns.append(final_return)
        
        win_rate = (np.array(trade_returns) > 0).mean() * 100 if trade_returns else 0
        
        # Calculate average win/loss
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r < 0]
        avg_win = np.mean(wins) * 100 if wins else 0
        avg_loss = np.mean(losses) * 100 if losses else 0

        return {
            "Total Return (%)": round(total_return, 2),
            "Annualized Return (%)": round(annualized_return, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown (%)": round(max_dd, 2),
            "Number of Trades": int(trades),
            "Win Rate (%)": round(win_rate, 2),
            "Avg Win (%)": round(avg_win, 2),
            "Avg Loss (%)": round(avg_loss, 2),
            "Final NAV": round(float(nav.iloc[-1]), 2)
        }
    
    def plot_results(self, df_backtest, ticker='Stock', save_path=None):
        """Plot backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # NAV over time with buy/hold comparison
        axes[0].plot(df_backtest.index, df_backtest['nav'], 
                    label='Strategy NAV', linewidth=2, color='blue')
        axes[0].axhline(y=self.init_cash, color='gray', 
                       linestyle='--', label='Initial Capital', alpha=0.7)
        
        # Add buy-and-hold comparison
        first_price = df_backtest['price'].iloc[0]
        last_price = df_backtest['price'].iloc[-1]
        bh_return = (last_price / first_price - 1) * 100
        bh_nav = self.init_cash * (1 + bh_return/100)
        axes[0].axhline(y=bh_nav, color='orange', linestyle=':', 
                       label=f'Buy & Hold ({bh_return:.1f}%)', alpha=0.7)
        
        axes[0].set_title(f'{ticker} - Portfolio Value Over Time', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('NAV ($)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Position sizes
        axes[1].fill_between(df_backtest.index, 0, df_backtest['shares'], 
                             where=(df_backtest['shares'] > 0), 
                             color='green', alpha=0.3, label='Long')
        axes[1].fill_between(df_backtest.index, 0, df_backtest['shares'], 
                             where=(df_backtest['shares'] < 0), 
                             color='red', alpha=0.3, label='Short')
        axes[1].set_title('Position Sizes', fontsize=12)
        axes[1].set_ylabel('Shares')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Price with trade signals
        axes[2].plot(df_backtest.index, df_backtest['price'], 
                    label='Price', color='black', alpha=0.6, linewidth=1.5)
        
        # Mark actual trades (where signal changed)
        trades = df_backtest[df_backtest['signal_changed']]
        buy_trades = trades[trades['signal'] > 0]
        sell_trades = trades[trades['signal'] < 0]
        exit_trades = trades[trades['signal'] == 0]
        
        if len(buy_trades) > 0:
            axes[2].scatter(buy_trades.index, buy_trades['price'], 
                           c='green', marker='^', s=100, 
                           label='Buy', zorder=5, edgecolors='darkgreen')
        if len(sell_trades) > 0:
            axes[2].scatter(sell_trades.index, sell_trades['price'], 
                           c='red', marker='v', s=100, 
                           label='Sell/Short', zorder=5, edgecolors='darkred')
        if len(exit_trades) > 0:
            axes[2].scatter(exit_trades.index, exit_trades['price'], 
                           c='gray', marker='x', s=100, 
                           label='Exit', zorder=5)
        
        axes[2].set_title('Price and Trade Execution', fontsize=12)
        axes[2].set_ylabel('Price ($)')
        axes[2].set_xlabel('Date')
        axes[2].legend(loc='upper left')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig


# Helper function to convert prediction results to proper signals
def create_signals_from_predictions(results_df):
    """
    Convert prediction results to discrete trading signals.
    
    Parameters:
    -----------
    results_df : DataFrame with 'signal' column containing 1, -1, or 0
    
    Returns:
    --------
    signals : Series with proper signal values
    confidence : Series with confidence scores
    """
    signals = results_df['signal'].copy()
    confidence = results_df.get('confidence', pd.Series(1.0, index=signals.index))
    
    return signals, confidence