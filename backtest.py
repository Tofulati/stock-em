import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AdvancedBacktest:
    """
    Advanced backtesting with:
    - Long/short positions
    - Position sizing based on confidence
    - Transaction costs
    - Risk management (stop-loss, position limits)
    """
    def __init__(
        self, 
        init_cash=10000,
        max_position_pct=0.3,
        transaction_cost=0.001,  # 0.1% per trade
        stop_loss_pct=0.05,      # 5% stop loss
        enable_shorting=True
    ):
        self.init_cash = init_cash
        self.max_position_pct = max_position_pct
        self.transaction_cost = transaction_cost
        self.stop_loss_pct = stop_loss_pct
        self.enable_shorting = enable_shorting
        
    def run(self, df_price, signals, confidence=None):
        """
        Run backtest with long/short positions
        
        Args:
            df_price: DataFrame with 'Open', 'Close', 'High', 'Low'
            signals: Series with values in {-1, 0, 1}
                     -1 = short, 0 = neutral/hold, 1 = long
            confidence: Optional Series with confidence [0, 1]
        """
        cash = self.init_cash
        shares = 0.0
        entry_price = None
        position_type = None  # 'long' or 'short'
        
        history = []
        
        for date in signals.index:
            if date not in df_price.index:
                continue
                
            signal = signals.loc[date]
            conf = confidence.loc[date] if confidence is not None else 1.0
            
            # Get prices
            open_price = df_price.loc[date]['Open']
            close_price = df_price.loc[date]['Close']
            high_price = df_price.loc[date]['High'] if 'High' in df_price.columns else close_price
            low_price = df_price.loc[date]['Low'] if 'Low' in df_price.columns else close_price
            
            # Check stop-loss
            if shares != 0 and entry_price is not None:
                if position_type == 'long':
                    # Stop loss for long: if price drops below entry - stop_loss_pct
                    if low_price < entry_price * (1 - self.stop_loss_pct):
                        # Execute stop loss at stop price
                        stop_price = entry_price * (1 - self.stop_loss_pct)
                        proceeds = shares * stop_price * (1 - self.transaction_cost)
                        cash += proceeds
                        shares = 0.0
                        entry_price = None
                        position_type = None
                        # print(f"[{date}] STOP LOSS triggered (long) at {stop_price:.2f}")
                        
                elif position_type == 'short':
                    # Stop loss for short: if price rises above entry + stop_loss_pct
                    if high_price > entry_price * (1 + self.stop_loss_pct):
                        # Close short position at stop price
                        stop_price = entry_price * (1 + self.stop_loss_pct)
                        # For shorts: we need to buy back shares at higher price (loss)
                        cost = abs(shares) * stop_price * (1 + self.transaction_cost)
                        cash -= cost
                        shares = 0.0
                        entry_price = None
                        position_type = None
                        # print(f"[{date}] STOP LOSS triggered (short) at {stop_price:.2f}")
            
            # Trading logic based on signal
            if signal > 0:  # BUY/LONG signal
                if shares < 0:  # Close short position first
                    # To close short: buy back shares
                    cost = abs(shares) * open_price * (1 + self.transaction_cost)
                    cash -= cost
                    shares = 0.0
                    entry_price = None
                    position_type = None
                
                # Open or add to long position
                if shares == 0 and cash > 0:
                    # Position sizing based on confidence
                    position_value = cash * self.max_position_pct * conf
                    new_shares = position_value / open_price
                    cost = new_shares * open_price * (1 + self.transaction_cost)
                    
                    if cost <= cash:
                        cash -= cost
                        shares = new_shares
                        entry_price = open_price
                        position_type = 'long'
                        
            elif signal < 0:  # SELL/SHORT signal
                if not self.enable_shorting:
                    # Just close long positions if shorting disabled
                    if shares > 0:
                        proceeds = shares * open_price * (1 - self.transaction_cost)
                        cash += proceeds
                        shares = 0.0
                        entry_price = None
                        position_type = None
                else:
                    # Close long position first if exists
                    if shares > 0:
                        proceeds = shares * open_price * (1 - self.transaction_cost)
                        cash += proceeds
                        shares = 0.0
                        entry_price = None
                        position_type = None
                    
                    # Open short position (limited by available cash as collateral)
                    if shares == 0 and cash > 0:
                        # Short position sizing (use cash as collateral)
                        position_value = cash * self.max_position_pct * conf
                        new_shares = position_value / open_price
                        
                        # For short: we receive cash from selling borrowed shares
                        # But we need collateral, so track shares as negative
                        proceeds = new_shares * open_price * (1 - self.transaction_cost)
                        cash += proceeds
                        shares = -new_shares  # Negative = short
                        entry_price = open_price
                        position_type = 'short'
                        
            elif signal == 0:  # HOLD/NEUTRAL - close all positions
                if shares > 0:  # Close long
                    proceeds = shares * open_price * (1 - self.transaction_cost)
                    cash += proceeds
                    shares = 0.0
                    entry_price = None
                    position_type = None
                elif shares < 0:  # Close short
                    cost = abs(shares) * open_price * (1 + self.transaction_cost)
                    cash -= cost
                    shares = 0.0
                    entry_price = None
                    position_type = None
            
            # Calculate NAV (Net Asset Value)
            if shares > 0:  # Long position
                position_value = shares * close_price
                nav = cash + position_value
            elif shares < 0:  # Short position
                # For short: NAV = cash - (current value of borrowed shares)
                # When we shorted, we got cash. Now we owe shares worth current price.
                borrowed_value = abs(shares) * close_price
                nav = cash - borrowed_value
            else:
                nav = cash
            
            history.append({
                'date': date,
                'cash': cash,
                'shares': shares,
                'position_type': position_type if shares != 0 else 'none',
                'position_value': shares * close_price if shares != 0 else 0,
                'nav': nav,
                'signal': signal,
                'confidence': conf
            })
        
        df_result = pd.DataFrame(history).set_index('date')
        return df_result
    
    def calculate_metrics(self, df_backtest):
        """Calculate performance metrics"""
        nav_series = df_backtest['nav']
        returns = nav_series.pct_change().dropna()
        
        # Total return
        total_return = (nav_series.iloc[-1] / nav_series.iloc[0] - 1) * 100
        
        # Annualized return (assuming 252 trading days)
        n_days = len(nav_series)
        years = n_days / 252
        annualized_return = ((nav_series.iloc[-1] / nav_series.iloc[0]) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        n_trades = (df_backtest['signal'].diff() != 0).sum()
        
        metrics = {
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Sharpe Ratio': sharpe,
            'Max Drawdown (%)': max_drawdown,
            'Final NAV': nav_series.iloc[-1],
            'Number of Trades': n_trades,
            'Volatility (%)': returns.std() * np.sqrt(252) * 100
        }
        
        return metrics
    
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