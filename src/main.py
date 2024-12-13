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

# %%
# # Calculate bsp_avg
data_clean.loc[:, 'bsp_avg'] = data_clean.filter(like='ob_10_p_').mean(axis=1)

# Set rolling window size
X = 1000

# Calculate min_bsp
data_clean.loc[:, 'min_bsp'] = data_clean.loc[:, 'ob_10_p_l'].rolling(window=X, min_periods=1).min()

# Detect points below min_bsp
data_clean.loc[:, 'below_min'] = data_clean.loc[:, 'ob_10_p_l'] <= data_clean.loc[:, 'min_bsp']

# Extract points below min_bsp
points_below_min = data_clean[data_clean.loc[:, 'below_min']]
print(f"Number of points below min: {len(points_below_min)}")

# %%
def filter_points(data, min_gap):
    # Ensure data is an independent DataFrame
    data = data.copy()

    data.loc[:, 'filterOne'] = False
    last_valid_time = -np.inf

    for idx, row in data.iterrows():
        if row['below_min']:
            if (row['ts'] - last_valid_time) >= min_gap * 60:
                data.at[idx, 'filterOne'] = True
                last_valid_time = row['ts']

    return data

# Minimum gap parameter
min_gap = 2

# Filter points
points_below_min = filter_points(points_below_min, min_gap)

# %%
def calculate_percentage_change(current_price, past_price):
    return (current_price - past_price) / past_price * 100

def calculate_trailing_exit(entry_price, prices, trailing_stop_loss, stop_loss, fixed_profit):
    max_price = entry_price
    for price in prices:
        if price > max_price:
            max_price = price

        if fixed_profit != 0 and price >= entry_price * (1 + fixed_profit / 100):
            return {'price': price, 'stop_loss_hit': False}

        if price > entry_price and price <= max_price * (1 - trailing_stop_loss / 100):
            return {'price': price, 'stop_loss_hit': False}

        if price <= entry_price * (1 - stop_loss / 100):
            return {'price': entry_price * (1 - stop_loss / 100), 'stop_loss_hit': True}

    return {'price': prices[-1], 'stop_loss_hit': False}

def calculate_fixed_exit(entry_price, prices, fixed_profit, fixed_loss):
    for price in prices:
        if price >= entry_price * (1 + fixed_profit / 100):
            return {'price': price, 'stop_loss_hit': False}

        if price <= entry_price * (1 - fixed_loss / 100):
            return {'price': entry_price * (1 - fixed_loss / 100), 'stop_loss_hit': True}

    return {'price': prices[-1], 'stop_loss_hit': False}

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
    show_chart: bool = True
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
        entry_ts = point['ts']
        # Initialize variables to avoid unbound issues
        transaction_status = None
        price_change_percent = None
        exit_trail_price = None
        result_tsl = None
        stop_loss_tsl = None
        exit_fixed_price = None
        result_fixed = None
        stop_loss_fixed = None

        # Determine entry price
        entry_price = (point['l'] + point['h']) / 2 if use_avg_price else point['l']

        # Check optional condition for past percentage drop
        if past_interval_percentage != 0 and past_percentage_min_dropdown != 0:
            past_ts = entry_ts - past_interval_percentage * 60
            past_prices = data_clean.loc[(data_clean['ts'] >= past_ts) & (data_clean['ts'] < entry_ts), 'h']
            max_past_price = past_prices.max() if not past_prices.empty else None # type: ignore

            if max_past_price is not None:
                drop_percent = calculate_percentage_change(entry_price, max_past_price)
                if drop_percent > past_percentage_min_dropdown:
                    transaction_status = "Price drop not enough"

        # if transaction_status is None:
        # Calculate price change percentage
        past_ts = entry_ts - past_interval_percentage * 60
        past_price = data_clean.loc[(data_clean['ts'] - past_ts).abs().idxmin(), 'l']
        price_change_percent = calculate_percentage_change(entry_price, past_price)
        # Get prices after entry
        prices_after_entry = data_clean.loc[data_clean['ts'] >= entry_ts, 'h'].values # type: ignore
        # Trailing stop-loss exit
        trail_exit = calculate_trailing_exit(entry_price, prices_after_entry, Y_trail_profit, Y_stop_loss, Y_trail_fixed_profit)
        exit_trail_price = trail_exit['price'] if 'price' in trail_exit else None
        stop_loss_tsl = trail_exit.get('stop_loss_hit', False)
        result_tsl = calculate_percentage_change(exit_trail_price, entry_price) if exit_trail_price else -Y_stop_loss
        # Fixed profit/loss exit
        fixed_exit = calculate_fixed_exit(entry_price, prices_after_entry, Y_fixed_profit, Y_fixed_loss)
        exit_fixed_price = fixed_exit['price'] if 'price' in fixed_exit else None
        stop_loss_fixed = fixed_exit.get('stop_loss_hit', False)
        result_fixed = calculate_percentage_change(exit_fixed_price, entry_price) if exit_fixed_price else -Y_fixed_loss
        # Update cumulative results
        cumulative_in_trade_result_tsl += result_tsl if transaction_status is None else 0
        cumulative_in_trade_result_fixed += result_fixed if transaction_status is None else 0
        cumulative_all_result_tsl += result_tsl
        cumulative_all_result_fixed += result_fixed
        # else:
        #     # Skip the trade, no changes to cumulative results
        #     price_change_percent = None
        #     exit_trail_price = None
        #     result_tsl = None
        #     stop_loss_tsl = None
        #     exit_fixed_price = None
        #     result_fixed = None

        # Add transaction details
        transactions.append({
            'entry_ts': entry_ts,
            'entry_price': entry_price,
            'price_change_percent': price_change_percent,
            'exit_trail_price': exit_trail_price,
            'result_tsl': result_tsl,
            'stop_loss_tsl': stop_loss_tsl,
            'exit_fixed_price': exit_fixed_price,
            'result_fixed': result_fixed,
            'stop_loss_fixed': stop_loss_fixed,
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
transactions = trade_simulation(points_below_min, data_clean, past_interval_percentage=120, past_percentage_min_dropdown=-2.5,
                                 Y_trail_profit=5, Y_stop_loss=10, Y_trail_fixed_profit=6, Y_fixed_profit=6, Y_fixed_loss=5)

# %%
transactions.to_json("transactions.json", orient='records', indent=4)
print(transactions.tail(1)[['cumulative_in_trade_result_tsl', 'cumulative_in_trade_result_fixed', 'cumulative_all_result_tsl', 'cumulative_all_result_fixed']])

