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

def trade_simulation(points_below_min, data_clean, past_interval_percentage=0, past_percentage_min_dropdown: float =0,
                     Y_trail_profit=3, Y_stop_loss=5, Y_trail_fixed_profit=0, Y_fixed_profit=10, Y_fixed_loss=5,
                     use_avg_price=False, show_chart=True):
    transactions = []
    cumulative_result_tsl = 0
    cumulative_result_fixed = 0

    for _, point in points_below_min.iterrows():
        entry_ts = point['ts']
        entry_price = (point['l'] + point['h']) / 2 if use_avg_price else point['l']

        if past_interval_percentage != 0 and past_percentage_min_dropdown != 0:
            past_ts = entry_ts - past_interval_percentage * 60
            past_prices = data_clean.loc[(data_clean['ts'] >= past_ts) & (data_clean['ts'] < entry_ts), 'h']
            max_past_price = past_prices.max() if not past_prices.empty else None

            if max_past_price is not None:
                drop_percent = calculate_percentage_change(entry_price, max_past_price)
                if drop_percent > past_percentage_min_dropdown:
                    continue

        past_ts = entry_ts - past_interval_percentage * 60
        past_price = data_clean.loc[(data_clean['ts'] - past_ts).abs().idxmin(), 'l']
        price_change_percent = calculate_percentage_change(entry_price, past_price)

        prices_after_entry = data_clean.loc[data_clean['ts'] >= entry_ts, 'h'].values

        trail_exit = calculate_trailing_exit(entry_price, prices_after_entry, Y_trail_profit, Y_stop_loss, Y_trail_fixed_profit)
        fixed_exit = calculate_fixed_exit(entry_price, prices_after_entry, Y_fixed_profit, Y_fixed_loss)

        result_tsl = calculate_percentage_change(trail_exit['price'], entry_price)
        result_fixed = calculate_percentage_change(fixed_exit['price'], entry_price)

        cumulative_result_tsl += result_tsl
        cumulative_result_fixed += result_fixed

        transactions.append({
            'entry_ts': entry_ts,
            'entry_price': entry_price,
            'price_change_percent': price_change_percent,
            'exit_trail_price': trail_exit['price'],
            'result_tsl': result_tsl,
            'exit_fixed_price': fixed_exit['price'],
            'result_fixed': result_fixed,
            'cumulative_result_tsl': cumulative_result_tsl,
            'cumulative_result_fixed': cumulative_result_fixed
        })

    if show_chart:
        transactions_df = pd.DataFrame(transactions)
        plt.figure(figsize=(12, 6))
        plt.plot(transactions_df['entry_ts'], transactions_df['cumulative_result_tsl'], label='TSL Strategy', color='blue')
        plt.plot(transactions_df['entry_ts'], transactions_df['cumulative_result_fixed'], label='Fixed Strategy', color='red')
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

transactions.to_json("transactions.json", orient='records', indent=4)
print(transactions.tail(1)[['cumulative_result_tsl', 'cumulative_result_fixed']])

# %%
