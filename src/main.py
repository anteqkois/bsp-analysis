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

# %%
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

def calculate_trailing_exit(
    entry_row, rows, trailing_stop_loss, stop_loss, fixed_profit, 
    use_trailing_stop_loss=False, use_stop_loss=False, use_fixed_profit=False,
    use_max_bsp_exit=False, points_above_max=None, price_accessor='avg'
):
    entry_price = entry_row[price_accessor]
    entry_ts = entry_row['ts']

    # Pre-filter rows to improve lookup
    rows_after_entry = rows[rows['ts'] > entry_ts]

    # Handle max BSP exit
    if use_max_bsp_exit and points_above_max is not None:
        points_above_max_filtered = points_above_max[points_above_max['ts'] > entry_ts]
        if not points_above_max_filtered.empty:
            max_bsp_ts = points_above_max_filtered.iloc[0]['ts']
            rows_after_entry = rows_after_entry[rows_after_entry['ts'] <= max_bsp_ts]
        else:
            use_max_bsp_exit = False

    # Initialize exit list
    exits = []

    # Fixed Profit Exit
    if use_fixed_profit:
        fixed_profit_exit = rows_after_entry[rows_after_entry[price_accessor] >= entry_price * (1 + fixed_profit / 100)]
        if not fixed_profit_exit.empty:
            row = fixed_profit_exit.iloc[0]
            exits.append({'price': row[price_accessor], 'ts': row['ts'], 'exit_reason': 'Fixed Profit'})

    # Trailing Stop Loss Exit
    if use_trailing_stop_loss:
        # Filter rows where price exceeds entry price (required for valid TSL activation)
        rows_above_entry = rows_after_entry[rows_after_entry[price_accessor] > entry_price]
        if not rows_above_entry.empty:
            max_price = rows_above_entry[price_accessor].cummax()
            tsl_exit = rows_above_entry[rows_above_entry[price_accessor] <= max_price * (1 - trailing_stop_loss / 100)]
            if not tsl_exit.empty:
                row = tsl_exit.iloc[0]
                exits.append({'price': row[price_accessor], 'ts': row['ts'], 'exit_reason': 'TSL'})

    # Stop Loss Exit
    if use_stop_loss:
        stop_loss_exit = rows_after_entry[rows_after_entry[price_accessor] <= entry_price * (1 - stop_loss / 100)]
        if not stop_loss_exit.empty:
            row = stop_loss_exit.iloc[0]
            exits.append({'price': row[price_accessor], 'ts': row['ts'], 'exit_reason': 'Stop Loss'})

    # Select the earliest exit based on timestamp
    if exits:
        earliest_exit = min(exits, key=lambda x: x['ts'])
        return earliest_exit

    # Default: No exit found
    return {'price': entry_price, 'ts': 0, 'exit_reason': None}

def calculate_fixed_exit(
    entry_row, rows, fixed_profit, fixed_loss, use_stop_loss=False, use_fixed_profit=False,
    use_max_bsp_exit=False, points_above_max=None, price_accessor='avg'
):
    entry_price = entry_row[price_accessor]
    entry_ts = entry_row['ts']

    # Pre-filter rows for faster lookup
    rows_after_entry = rows[rows['ts'] > entry_ts]

    # Handle Max BSP Exit
    if use_max_bsp_exit and points_above_max is not None:
        points_above_max_filtered = points_above_max[points_above_max['ts'] > entry_ts]
        if not points_above_max_filtered.empty:
            point_bsp_exit_ts = points_above_max_filtered.iloc[0]['ts']
            rows_after_entry = rows_after_entry[rows_after_entry['ts'] <= point_bsp_exit_ts]
        else:
            use_max_bsp_exit = False

    # Collect potential exits
    exits = []

    # Fixed Profit Exit
    if use_fixed_profit:
        fixed_profit_exit = rows_after_entry[rows_after_entry[price_accessor] >= entry_price * (1 + fixed_profit / 100)]
        if not fixed_profit_exit.empty:
            row = fixed_profit_exit.iloc[0]
            exits.append({'price': row[price_accessor], 'ts': row['ts'], 'exit_reason': 'Fixed Profit'})

    # Stop Loss Exit
    if use_stop_loss:
        stop_loss_exit = rows_after_entry[rows_after_entry[price_accessor] <= entry_price * (1 - fixed_loss / 100)]
        if not stop_loss_exit.empty:
            row = stop_loss_exit.iloc[0]
            exits.append({'price': row[price_accessor], 'ts': row['ts'], 'exit_reason': 'Stop Loss'})

    # Max BSP Exit
    if use_max_bsp_exit and not rows_after_entry.empty:
        row = rows_after_entry.iloc[0]  # First row within Max BSP time constraint
        exits.append({'price': row[price_accessor], 'ts': row['ts'], 'exit_reason': 'Max BSP Crossed'})

    # Select the earliest exit based on timestamp
    if exits:
        earliest_exit = min(exits, key=lambda x: x['ts'])
        return earliest_exit

    # Default: No exit found
    return {'price': entry_price, 'ts': 0, 'exit_reason': None}

# %%
# Load data
data = pd.read_json("../data/data-crypto-TAOUSDT.json")
data.drop(columns=['ticker'], inplace=True)

# %%
# Handle missing values
data_clean = data.copy().dropna()
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
    use_stop_loss: bool = False,
    use_trailing_stop_loss: bool = False,
    use_fixed_profit: bool = False,
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

    rows_after_entry = data_clean.copy()

    for _, point in points_below_min.iterrows():
        print(f"Transaction: {point['ts']}")

        entry_ts = point['ts']
        # Initialize variables to avoid unbound issues
        transaction_status = None
        price_change_percent = None
        result_tsl = None
        result_fixed = None

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
        
        # Get rowsd after entry
        rows_after_entry = rows_after_entry.loc[data_clean['ts'] >= entry_ts]
        
        # Trailing stop-loss exit
        tsl_exit = calculate_trailing_exit(point, rows_after_entry, Y_trail_profit, Y_stop_loss, Y_trail_fixed_profit, use_trailing_stop_loss, use_stop_loss, use_fixed_profit,  use_max_bsp_exit, points_above_max, 'l')
        # TODO change 0 to None ???
        result_tsl = calculate_percentage_change(tsl_exit['price'], entry_price) if tsl_exit['price'] else 0
        
        # Fixed profit/loss exit
        fixed_exit = calculate_fixed_exit(point, rows_after_entry, Y_fixed_profit, Y_fixed_loss, use_stop_loss, use_fixed_profit, use_max_bsp_exit, points_above_max, 'l')
        result_fixed = calculate_percentage_change(fixed_exit['price'], entry_price) if fixed_exit['price'] else 0
        # fixed_exit = {
        #     'price': 1,
        #     'ts': 1, 
        #     'exit_reason': ''
        # }
        # result_fixed = 0
        
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
            'exit_tsl_price': tsl_exit['price'],
            'result_tsl': result_tsl,
            'tsl_exit_ts': tsl_exit['ts'],
            'tsl_exit_reason': tsl_exit['exit_reason'],
            'exit_fixed_price': fixed_exit['price'],
            'result_fixed': result_fixed,
            'fixed_exit_ts': fixed_exit['ts'],
            'fixed_exit_reason': fixed_exit['exit_reason'],
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
                                 Y_trail_profit=5, Y_stop_loss=10, Y_trail_fixed_profit=6, Y_fixed_profit=6, Y_fixed_loss=5, use_stop_loss = True,
                                 use_trailing_stop_loss = True, use_fixed_profit = True, use_max_bsp_exit = False, points_above_max=points_above_max)

# %%
transactions_without_bsp_max.to_json("transactions_without_bsp_max.json", orient='records', indent=4)
print(transactions_without_bsp_max.tail(1)[['cumulative_in_trade_result_tsl', 'cumulative_in_trade_result_fixed', 'cumulative_all_result_tsl', 'cumulative_all_result_fixed']])

# %%
# Trades with BSP max
transactions_with_bsp_max = trade_simulation(points_below_min[:10], data_clean, past_interval_percentage=120, past_percentage_min_dropdown=-2.5,
                                 Y_trail_profit=5, Y_stop_loss=10, Y_trail_fixed_profit=6, Y_fixed_profit=6, Y_fixed_loss=5, use_stop_loss = False,
                                 use_trailing_stop_loss = False, use_fixed_profit = False, use_max_bsp_exit = True, points_above_max=points_above_max)

# %%
transactions_with_bsp_max.to_json("transactions_with_bsp_max.json", orient='records', indent=4)
print(transactions_with_bsp_max.tail(1)[['cumulative_in_trade_result_tsl', 'cumulative_in_trade_result_fixed', 'cumulative_all_result_tsl', 'cumulative_all_result_fixed']])

# %%
