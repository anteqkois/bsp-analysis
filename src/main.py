# %%
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install numpy pandas scipy matplotlib # type: ignore
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import Optional

# Settings
interval = '1m'

# Helpers
def get_interval_coefficient(interval):
    if interval == '1m' or interval == 'block':
        return 60
    elif interval == '5m':
        return 60 * 5
    elif interval == '10m':
        return 60 * 10
    elif interval == '15m':
        return 60 * 15
    elif interval == '1h' or interval == 'hour':
        return 60 * 60
    elif interval == '2h':
        return 60 * 60 * 2
    elif interval == '4h':
        return 60 * 60 * 4
    elif interval == '12h':
        return 60 * 60 * 12
    elif interval == '1d' or interval == 'day':
        return 60 * 60 * 24
    elif interval == '3d':
        return 60 * 60 * 72
    elif interval == '1w':
        return 60 * 60 * 24 * 7
    elif interval == '1M':
        return 60 * 60 * 24 * 31
    else:
        return 60

def calculate_percentage_change(current_price, past_price):
    return (current_price - past_price) / past_price * 100


def calculate_trailing_exit(entry_price, prices, trailing_stop_loss, stop_loss, fixed_profit, 
                             use_max_bsp_exit=False, points_above_max=None):
    max_price = entry_price
    for price in prices:
        if price > max_price:
            max_price = price

        if fixed_profit != 0 and price >= entry_price * (1 + fixed_profit / 100):
            return {'price': price, 'exit_reason': 'Fixed Profit'}

        if price > entry_price and price <= max_price * (1 - trailing_stop_loss / 100):
            return {'price': price, 'exit_reason': 'TSL'}

        if price <= entry_price * (1 - stop_loss / 100):
            return {'price': entry_price * (1 - stop_loss / 100), 'exit_reason': 'Stop Loss'}

        if use_max_bsp_exit and points_above_max is not None:
            # Check if max_bsp was crossed after entry
            max_bsp_crossed = points_above_max[(points_above_max['ts'] >= entry_price) & (points_above_max['ts'] <= price)]
            if not max_bsp_crossed.empty:
                return {'price': max_bsp_crossed.iloc[0]['h'], 'exit_reason': 'Max BSP Crossed'}

    return {'price': prices[-1], 'exit_reason': None}

def calculate_fixed_exit(entry_price, prices, fixed_profit, fixed_loss, 
                         use_max_bsp_exit=False, points_above_max=None):
    for price in prices:
        if price >= entry_price * (1 + fixed_profit / 100):
            return {'price': price, 'exit_reason': 'Fixed Profit'}

        if price <= entry_price * (1 - fixed_loss / 100):
            return {'price': entry_price * (1 - fixed_loss / 100), 'exit_reason': 'Stop Loss'}

        if use_max_bsp_exit and points_above_max is not None:
            # Check if max_bsp was crossed after entry
            max_bsp_crossed = points_above_max[(points_above_max['ts'] >= entry_price) & (points_above_max['ts'] <= price)]
            if not max_bsp_crossed.empty:
                return {'price': max_bsp_crossed.iloc[0]['h'], 'exit_reason': 'Max BSP Crossed'}

    return {'price': prices[-1], 'exit_reason': None}

# %%
# Load data
data = pd.read_json("../data/data-crypto-TAOUSDT.json")
data.drop(columns=['ticker'], inplace=True)
# print(data.info())  # Logs column names, types, and null counts
# print(data.head())  # Shows the first few rows
# print(data.describe())  # Provides statistical summary for numeric columns

# %%
# Handle missing values
if data.isna().any().any():
    data_clean = data.dropna().copy()  # Use .copy() to create an explicit copy
else:
    data_clean = data.copy()
print(f"Number of records after clean: {len(data_clean)}")

# %%
# # Calculate bsp_avg
# data_clean.loc[:, 'bsp_avg'] = data_clean.filter(like='ob_10_p_').mean(axis=1)

# Set rolling window size
X = 1000

# Calculate min_bsp, and max_bsp
data_clean.loc[:, 'min_bsp'] = data_clean['ob_10_p_l'].rolling(window=X, min_periods=1).min()
data_clean.loc[:, 'max_bsp'] = data_clean['ob_10_p_h'].rolling(window=X, min_periods=1).max()

# Detect points below min_bsp, and above max_bsp
data_clean.loc[:, 'below_min'] = data_clean['ob_10_p_l'] <= data_clean['min_bsp']
data_clean.loc[:, 'above_max'] = data_clean['ob_10_p_h'] >= data_clean['max_bsp']

# Extract points below min_bsp and points above max
points_below_min = data_clean[data_clean['below_min']]
print(f"Number of points below min: {len(points_below_min)}")
points_above_max = data_clean[data_clean['above_max']]
print(f"Number of points above max: {len(points_above_max)}")

# %%
def filter_points(data, min_interval_gap):
    data = data.copy()

    data.loc[:, 'intervalGapFilter'] = False
    last_valid_time = -np.inf

    for idx, row in data.iterrows():
        if row['below_min']:
            if (row['ts'] - last_valid_time) >= min_interval_gap * get_interval_coefficient(interval):
                data.at[idx, 'intervalGapFilter'] = True
                last_valid_time = row['ts']

    return data

# Minimum gap parameter
min_interval_gap = 2

# Filter points
points_below_min = filter_points(points_below_min, min_interval_gap)

# %%
def trade_simulation(
    points_below_min: pd.DataFrame,
    data_clean: pd.DataFrame,
    past_interval_percentage: int = 0,
    past_percentage_min_dropdown: float = 0,
    Y_trail_profit: float = 3,
    Y_stop_loss: float = 5,
    Y_trail_fixed_profit: float = 0,
    Y_fixed_profit: float = 10,
    Y_fixed_loss: float = 5,
    use_avg_price: bool = False,
    show_chart: bool = True,
    use_max_bsp_exit: bool = False,
    points_above_max: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Simulates trading based on provided conditions and parameters.

    Args:
        points_below_min (pd.DataFrame): Points where conditions trigger a potential trade.
        data_clean (pd.DataFrame): Full dataset of market data.
        past_interval_percentage (int): Time interval for past percentage drop check (minutes).
        past_percentage_min_dropdown (float): Minimum percentage drop required in past interval.
        Y_trail_profit (float): Trailing profit threshold (percentage).
        Y_stop_loss (float): Stop loss threshold (percentage).
        Y_trail_fixed_profit (float): Fixed profit threshold for trailing strategy (percentage).
        Y_fixed_profit (float): Fixed profit target (percentage).
        Y_fixed_loss (float): Fixed loss threshold (percentage).
        use_avg_price (bool): Whether to use the average price for entry.
        show_chart (bool): Whether to show a cumulative results chart.

    Returns:
        pd.DataFrame: Transactions with details of each trade.
    """
    transactions = []
    cumulative_in_trade_result_tsl = 0
    cumulative_in_trade_result_fixed = 0
    cumulative_all_result_tsl = 0
    cumulative_all_result_fixed = 0

    for _, point in points_below_min.iterrows():
        print(f"Number of points above max: {point['ts']}")
        entry_ts = point['ts']
        # Initialize variables to avoid unbound issues
        transaction_status = None
        price_change_percent = None
        exit_trail_price = None
        result_tsl = None
        tsl_exit_reason = None
        exit_fixed_price = None
        result_fixed = None
        fixed_exit_reason = None

        # Determine entry price
        entry_price = (point['l'] + point['h']) / 2 if use_avg_price else point['l']

        # Check optional condition for past percentage drop
        if past_interval_percentage != 0 and past_percentage_min_dropdown != 0:
            past_ts = entry_ts - past_interval_percentage * get_interval_coefficient(interval)
            past_prices = data_clean.loc[(data_clean['ts'] >= past_ts) & (data_clean['ts'] < entry_ts), 'h']
            max_past_price = past_prices.max() if not past_prices.empty else None # type: ignore

            if max_past_price is not None:
                drop_percent = calculate_percentage_change(entry_price, max_past_price)
                if drop_percent > past_percentage_min_dropdown:
                    transaction_status = "Price drop not enough"

        # Calculate price change percentage
        past_ts = entry_ts - past_interval_percentage * get_interval_coefficient(interval)
        past_price = data_clean.loc[(data_clean['ts'] - past_ts).abs().idxmin(), 'l']
        price_change_percent = calculate_percentage_change(entry_price, past_price)
        
        # Get prices after entry
        prices_after_entry = data_clean.loc[data_clean['ts'] >= entry_ts, 'h'].values # type: ignore
        
        # Trailing stop-loss exit
        trail_exit = calculate_trailing_exit(entry_price, prices_after_entry, Y_trail_profit, Y_stop_loss, Y_trail_fixed_profit, use_max_bsp_exit, points_above_max)
        exit_trail_price = trail_exit['price'] if 'price' in trail_exit else None
        tsl_exit_reason = trail_exit['exit_reason']
        result_tsl = calculate_percentage_change(exit_trail_price, entry_price) if exit_trail_price else -Y_stop_loss
        
        # Fixed profit/loss exit
        fixed_exit = calculate_fixed_exit(entry_price, prices_after_entry, Y_fixed_profit, Y_fixed_loss, use_max_bsp_exit, points_above_max)
        exit_fixed_price = fixed_exit['price'] if 'price' in fixed_exit else None
        fixed_exit_reason = fixed_exit['exit_reason']
        result_fixed = calculate_percentage_change(exit_fixed_price, entry_price) if exit_fixed_price else -Y_fixed_loss
        
        # Update cumulative results
        cumulative_in_trade_result_tsl += result_tsl if transaction_status is None else 0
        cumulative_in_trade_result_fixed += result_fixed if transaction_status is None else 0
        cumulative_all_result_tsl += result_tsl
        cumulative_all_result_fixed += result_fixed

        # Add transaction details
        transactions.append({
            'entry_ts': entry_ts,
            'entry_price': entry_price,
            'price_change_percent': price_change_percent,
            'exit_trail_price': exit_trail_price,
            'result_tsl': result_tsl,
            'tsl_exit_reason': tsl_exit_reason,
            'exit_fixed_price': exit_fixed_price,
            'result_fixed': result_fixed,
            'fixed_exit_reason': fixed_exit_reason,
            'cumulative_in_trade_result_tsl': cumulative_in_trade_result_tsl,
            'cumulative_in_trade_result_fixed': cumulative_in_trade_result_fixed,
            'cumulative_all_result_tsl': cumulative_all_result_tsl,
            'cumulative_all_result_fixed': cumulative_all_result_fixed,
            'transaction_status': transaction_status or "Trade"
        })

    # Plot cumulative results
    if show_chart:
        transactions_df = pd.DataFrame(transactions)
        plt.figure(figsize=(12, 6))
        plt.plot(transactions_df['entry_ts'], transactions_df['cumulative_in_trade_result_tsl'], label='TSL Strategy Traded', color='blue')
        plt.plot(transactions_df['entry_ts'], transactions_df['cumulative_in_trade_result_fixed'], label='Fixed Strategy Traded', color='red')
        plt.plot(transactions_df['entry_ts'], transactions_df['cumulative_all_result_tsl'], label='TSL Strategy All', color='blue', alpha=0.5)
        plt.plot(transactions_df['entry_ts'], transactions_df['cumulative_all_result_fixed'], label='Fixed Strategy All', color='red', alpha=0.5)
        plt.xlabel('Timestamp')
        plt.ylabel('Cumulative Result (%)')
        plt.title('Cumulative Results: TSL vs Fixed Strategy')
        plt.legend()
        plt.grid()
        plt.show()

    return pd.DataFrame(transactions)

# %%
# Trades without BSP max
transactions_without_bsp_max = trade_simulation(points_below_min, data_clean, past_interval_percentage=120, past_percentage_min_dropdown=-2.5,
                                 Y_trail_profit=5, Y_stop_loss=10, Y_trail_fixed_profit=6, Y_fixed_profit=6, Y_fixed_loss=5)

# %%
transactions_without_bsp_max.to_json("transactions_without_bsp_max.json", orient='records', indent=4)
print(transactions_without_bsp_max.tail(1)[['cumulative_in_trade_result_tsl', 'cumulative_in_trade_result_fixed', 'cumulative_all_result_tsl', 'cumulative_all_result_fixed']])


# %%
# Trades with BSP max
transactions_with_bsp_max = trade_simulation(points_below_min, data_clean, past_interval_percentage=120, past_percentage_min_dropdown=-2.5,
                                 Y_trail_profit=5, Y_stop_loss=10, Y_trail_fixed_profit=6, Y_fixed_profit=6, Y_fixed_loss=5, use_max_bsp_exit=True, points_above_max=points_above_max)

# %%
transactions_with_bsp_max.to_json("transactions_with_bsp_max.json", orient='records', indent=4)
print(transactions_with_bsp_max.tail(1)[['cumulative_in_trade_result_tsl', 'cumulative_in_trade_result_fixed', 'cumulative_all_result_tsl', 'cumulative_all_result_fixed']])

# %%
