# %%
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install numpy pandas scipy matplotlib # type: ignore
import numpy as np
import pandas as pd
import time
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

def calculate_trailing_exit(entry_row, rows, tsl_trailing_stop_loss:float=0, stop_loss:float=0, take_profit:float=0, points_above_max=None):
    entry_ts, entry_h, entry_l, entry_c = entry_row[['ts', 'h', 'l', 'c']]
    max_price = entry_h

    exit_points = []

    # Update max price vectorized
    rows['max_price'] = rows['h'].cummax()

    # Check for fixed profit exit condition
    if take_profit:
        fixed_profit_rows = rows[rows['h'] >= entry_c * (1 + take_profit / 100)]
        if not fixed_profit_rows.empty:
            first_fixed_profit_row = fixed_profit_rows.iloc[0]
            exit_points.append({'price': first_fixed_profit_row['h'], 'exit_reason': 'Fixed Profit', 'ts': first_fixed_profit_row['ts']})

    # Check for trailing stop loss exit condition
    if tsl_trailing_stop_loss:
        tsl_rows = rows[(rows['h'] > entry_c) & (rows['h'] <= rows['max_price'] * (1 - tsl_trailing_stop_loss / 100))]
        if not tsl_rows.empty:
            first_tsl_trailing_stop_loss_row = tsl_rows.iloc[0]
            exit_points.append({'price': first_tsl_trailing_stop_loss_row['h'], 'exit_reason': 'TSL', 'ts': first_tsl_trailing_stop_loss_row['ts']})

    # Check for stop loss exit condition
    if stop_loss:
        stop_loss_price = entry_c * (1 - stop_loss / 100)
        stop_loss_rows = rows[rows['l'] <= stop_loss_price]
        if not stop_loss_rows.empty:
            first_stop_loss_row = stop_loss_rows.iloc[0]
            exit_points.append({'price': stop_loss_price, 'exit_reason': 'Stop Loss', 'ts': first_stop_loss_row['ts']})

    # Check for Max BSP exit condition if enabled
    if points_above_max is not None:
        # Filter for points where `above_max` is True and `ts` > entry_ts
        filtered_points = points_above_max[(points_above_max['above_max'] == True) & (points_above_max['ts'] > entry_ts)]

        if not filtered_points.empty:
            first_bsp_point = filtered_points.iloc[0]  # Get the first point after entry_ts
            exit_points.append({'price': first_bsp_point['h'], 'exit_reason': 'Max BSP Crossed', 'ts': first_bsp_point['ts']})

    # If no exit condition is met, return the last row's price and timestamp
    if not exit_points:
        last_row = rows.iloc[-1]
        return {'price': last_row['c'], 'exit_reason': None, 'ts': last_row['ts']}
    
    # Otherwise, return the exit point with the earliest timestamp
    first_exit = min(exit_points, key=lambda x: x['ts'])
    return first_exit

def calculate_fixed_exit(entry_row, rows, take_profit:float=0, stop_loss:float=0, points_above_max=None):
    entry_ts, entry_h, entry_l, entry_c = entry_row[['ts', 'h', 'l', 'c']]
    
    exit_points = []

    # Check for the first row where fixed profit condition is met
    if take_profit:
        # Calculate the target price for the fixed profit condition
        target_price = entry_c * (1 + take_profit / 100)
        fixed_profit_rows = rows[rows['h'] >= target_price]
        if not fixed_profit_rows.empty:
            first_fixed_profit_row = fixed_profit_rows.iloc[0]
            exit_points.append({'price': first_fixed_profit_row['h'], 'exit_reason': 'Fixed Profit', 'ts': first_fixed_profit_row['ts']})

    # Check for the first row where stop loss condition is met
    if stop_loss:
        stop_loss_price = entry_c * (1 - stop_loss / 100)
        stop_loss_rows = rows[rows['l'] <= stop_loss_price]
        if not stop_loss_rows.empty:
            first_stop_loss_row = stop_loss_rows.iloc[0]
            exit_points.append({'price': stop_loss_price, 'exit_reason': 'Stop Loss', 'ts': first_stop_loss_row['ts']})

    # Check for Max BSP exit condition if enabled
    if points_above_max is not None:
        # Filter for points where `above_max` is True and `ts` > entry_ts
        filtered_points = points_above_max[(points_above_max['above_max'] == True) & (points_above_max['ts'] > entry_ts)]

        if not filtered_points.empty:
            first_bsp_point = filtered_points.iloc[0]  # Get the first point after entry_ts
            exit_points.append({'price': first_bsp_point['h'], 'exit_reason': 'Max BSP Crossed', 'ts': first_bsp_point['ts']})

    # If no exit condition is met, return the last row's price and timestamp
    if not exit_points:
        last_row = rows.iloc[-1]
        return {'price': last_row['c'], 'exit_reason': None, 'ts': last_row['ts']}
    
    # Otherwise, return the exit point with the earliest timestamp
    first_exit = min(exit_points, key=lambda x: x['ts'])
    return first_exit

# %%
# Load data
data = pd.read_json(f"../data/data-crypto-TAOUSDT-{interval}.json")
data.drop(columns=['ticker'], inplace=True)

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
# # Create the first plot for min_bsp (using ob_10_p_l) with increased width
# plt.figure(figsize=(24, 6))  # Increased width by 2-3x

# # Plot the BSP Metric (ob_10_p_l) line
# plt.plot(data_clean['ts'], data_clean['ob_10_p_l'], label="Metryka BSP (ob_10_p_l)", color="#4682B4")

# # Plot the min_bsp line (dashed)
# plt.plot(data_clean['ts'], data_clean['min_bsp'], label="Wartości minimum (min_bsp)", color="#DAA520", linestyle="--")

# # Plot the points below min_bsp
# plt.scatter(points_below_min['ts'], points_below_min['ob_10_p_l'], label="Punkty przecięcia", color="red", s=50)

# # Add labels and title
# plt.title("Punkty przecięcia metryki BSP z wartościami minimum")
# plt.xlabel("Timestamp")
# plt.ylabel("Wartość")
# plt.legend(title="Legend", loc="lower center", bbox_to_anchor=(0.5, -0.05), shadow=True, fancybox=True)

# # Show the plot
# plt.tight_layout()
# plt.show()

# # Create the second plot for max_bsp (using ob_10_p_h) with increased width
# plt.figure(figsize=(24, 6))  # Increased width by 2-3x

# # Plot the BSP Metric (ob_10_p_h) line
# plt.plot(data_clean['ts'], data_clean['ob_10_p_h'], label="Metryka BSP (ob_10_p_h)", color="#32CD32")

# # Plot the max_bsp line (solid)
# plt.plot(data_clean['ts'], data_clean['max_bsp'], label="Wartości maksimum (max_bsp)", color="#8A2BE2", linestyle="-")

# # Plot the points above max_bsp
# plt.scatter(points_above_max['ts'], points_above_max['ob_10_p_h'], label="Punkty przecięcia max", color="orange", s=50)

# # Add labels and title
# plt.title("Punkty przecięcia metryki BSP z wartościami maksimum")
# plt.xlabel("Timestamp")
# plt.ylabel("Wartość")
# plt.legend(title="Legend", loc="lower center", bbox_to_anchor=(0.5, -0.05), shadow=True, fancybox=True)

# # Show the plot
# plt.tight_layout()
# plt.show()

# # Create the third plot for both min_bsp and max_bsp on the same chart with increased width
# plt.figure(figsize=(24, 6))  # Increased width by 2-3x

# # Plot the BSP Metric (ob_10_p_l) line
# plt.plot(data_clean['ts'], data_clean['ob_10_p_l'], label="Metryka BSP (ob_10_p_l)", color="#4682B4")

# # Plot the min_bsp line (dashed)
# plt.plot(data_clean['ts'], data_clean['min_bsp'], label="Wartości minimum (min_bsp)", color="#DAA520", linestyle="--")

# # Plot the points below min_bsp
# plt.scatter(points_below_min['ts'], points_below_min['ob_10_p_l'], label="Punkty przecięcia min", color="red", s=50)

# # Plot the BSP Metric (ob_10_p_h) line
# plt.plot(data_clean['ts'], data_clean['ob_10_p_h'], label="Metryka BSP (ob_10_p_h)", color="#32CD32")

# # Plot the max_bsp line (solid)
# plt.plot(data_clean['ts'], data_clean['max_bsp'], label="Wartości maksimum (max_bsp)", color="#8A2BE2", linestyle="-")

# # Plot the points above max_bsp
# plt.scatter(points_above_max['ts'], points_above_max['ob_10_p_h'], label="Punkty przecięcia max", color="orange", s=50)

# # Add labels and title
# plt.title("Punkty przecięcia metryki BSP z wartościami minimum i maksimum")
# plt.xlabel("Timestamp")
# plt.ylabel("Wartość")
# plt.legend(title="Legend", loc="lower center", bbox_to_anchor=(0.5, -0.05), shadow=True, fancybox=True)

# # Show the plot
# plt.tight_layout()
# plt.show()

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
    tsl_trailing_stop_loss: float = 3,
    tsl_stop_loss: float = 5,
    tsl_take_profit: float = 0,
    fixed_take_profit: float = 10,
    fixed_stop_loss: float = 5,
    use_avg_price: bool = False,
    show_chart: bool = True,
    points_above_max: Optional[pd.DataFrame] = None,
    chart_title: str = 'Cumulative Results: TSL vs Fixed Strategy'
) -> pd.DataFrame:
    transactions = []
    cumulative_in_trade_result_tsl = 0
    cumulative_in_trade_result_fixed = 0
    cumulative_all_result_tsl = 0
    cumulative_all_result_fixed = 0

    data_after_entry = data_clean[['ts', 'h', 'l', 'c']]

    for index, point in points_below_min.iterrows():
        # if index % 500 == 0: # type: ignore
        #     print(f"Transaction: {point['ts']}, {index}")
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
        entry_price = (point['l'] + point['h']) / 2 if use_avg_price else point['c']

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
        # data_after_entry = data_clean.loc[data_clean['ts'] >= entry_ts, ['ts', 'h', 'l', 'c']]
        data_after_entry = data_after_entry.loc[data_clean['ts'] >= entry_ts]
        
        # Trailing stop-loss exit
        trail_exit = calculate_trailing_exit(point, data_after_entry, tsl_trailing_stop_loss, tsl_stop_loss, tsl_take_profit, points_above_max)
        exit_trail_price = trail_exit['price'] if 'price' in trail_exit else None
        tsl_exit_reason = trail_exit['exit_reason']
        result_tsl = calculate_percentage_change(exit_trail_price, entry_price) if exit_trail_price else -tsl_stop_loss
        
        # Fixed profit/loss exit
        fixed_exit = calculate_fixed_exit(point, data_after_entry, fixed_take_profit, fixed_stop_loss, points_above_max)
        exit_fixed_price = fixed_exit['price'] if 'price' in fixed_exit else None
        fixed_exit_reason = fixed_exit['exit_reason']
        result_fixed = calculate_percentage_change(exit_fixed_price, entry_price) if exit_fixed_price else -fixed_stop_loss
        
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
        # Assuming transactions is already a list or DataFrame containing the data
        transactions_df = pd.DataFrame(transactions)

        # Create the figure
        plt.figure(figsize=(12, 6))

        # Plot the different strategies
        plt.plot(transactions_df['entry_ts'], transactions_df['cumulative_in_trade_result_tsl'], label='TSL Strategy Traded', color='blue')
        plt.plot(transactions_df['entry_ts'], transactions_df['cumulative_in_trade_result_fixed'], label='Fixed Strategy Traded', color='red')
        plt.plot(transactions_df['entry_ts'], transactions_df['cumulative_all_result_tsl'], label='TSL Strategy All', color='blue', alpha=0.5)
        plt.plot(transactions_df['entry_ts'], transactions_df['cumulative_all_result_fixed'], label='Fixed Strategy All', color='red', alpha=0.5)

        # Adding the final results as annotations (last value in each series)
        final_tsl_traded = transactions_df['cumulative_in_trade_result_tsl'].iloc[-1]
        final_fixed_traded = transactions_df['cumulative_in_trade_result_fixed'].iloc[-1]
        final_tsl_all = transactions_df['cumulative_all_result_tsl'].iloc[-1]
        final_fixed_all = transactions_df['cumulative_all_result_fixed'].iloc[-1]

        # Annotate the last points with text
        plt.text(transactions_df['entry_ts'].iloc[-1], final_tsl_traded, f'TSL Traded: {final_tsl_traded:.2f}%', color='blue', fontsize=10, ha='left')
        plt.text(transactions_df['entry_ts'].iloc[-1], final_fixed_traded, f'Fixed Traded: {final_fixed_traded:.2f}%', color='red', fontsize=10, ha='left')
        plt.text(transactions_df['entry_ts'].iloc[-1], final_tsl_all, f'TSL All: {final_tsl_all:.2f}%', color='blue', fontsize=10, ha='left', alpha=0.5)
        plt.text(transactions_df['entry_ts'].iloc[-1], final_fixed_all, f'Fixed All: {final_fixed_all:.2f}%', color='red', fontsize=10, ha='left', alpha=0.5)

        # Labels, title, and grid
        plt.xlabel('Timestamp')
        plt.ylabel('Cumulative Result (%)')
        plt.title(chart_title)
        plt.legend()
        plt.grid()

        # Show the plot
        plt.show()

    return pd.DataFrame(transactions)

# %%
# # Trades without BSP max
start_time = time.time()
transactions_without_bsp_max = trade_simulation(points_below_min, data_clean, past_interval_percentage=120, past_percentage_min_dropdown=-2.5,
                                 tsl_trailing_stop_loss=5, tsl_stop_loss=10, tsl_take_profit=6, fixed_take_profit=6, fixed_stop_loss=5)
elapsed_time = time.time() - start_time
print(f"Time taken for transactions: {elapsed_time:.2f} seconds")

# # %%
# Trades with BSP max
transactions_with_bsp_max = trade_simulation(points_below_min, data_clean, past_interval_percentage=120, past_percentage_min_dropdown=-2.5,
                                 tsl_trailing_stop_loss=5, tsl_stop_loss=10, tsl_take_profit=6, fixed_take_profit=6, fixed_stop_loss=5,
                                points_above_max=points_above_max, chart_title='SL=False TSL=False TP=False BSP=True')

# # %%
# Trades with BSP max
transactions_with_bsp_max = trade_simulation(points_below_min, data_clean, past_interval_percentage=120, past_percentage_min_dropdown=-2.5,
                                 tsl_trailing_stop_loss=5, tsl_stop_loss=10, tsl_take_profit=8, fixed_take_profit=8, fixed_stop_loss=5,
                                points_above_max=points_above_max, chart_title='SL=True TSL=True TP=True BSP=True')
# %%
