import json
import numpy as np
import pandas as pd

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

def filter_min_bsp(data: pd.DataFrame, back_interval_amount: int):
    # Calculate min_bsp
    data.loc[:, 'min_bsp'] = data['ob_10_p_l'].rolling(window=back_interval_amount, min_periods=1).min()

    # Detect points below min_bsp, and above max_bsp
    data.loc[:, 'below_min'] = data['ob_10_p_l'] <= data['min_bsp']

    # Extract points below min_bsp and points above max
    points_below_min = data[data['below_min']]
    print(f"Number of points below min: {len(points_below_min)}")
    
    return data, points_below_min

def filter_max_bsp(data: pd.DataFrame, back_interval_amount: int):
    data.loc[:, 'max_bsp'] = data['ob_10_p_h'].rolling(window=back_interval_amount, min_periods=1).max()
    data.loc[:, 'above_max'] = data['ob_10_p_h'] >= data['max_bsp']

    # Extract points below min_bsp and points above max
    points_above_max = data[data['above_max']]
    print(f"Number of points above max: {len(points_above_max)}")
    
    return data, points_above_max

def filter_min_interval_gap(data, min_interval_gap: int, interval: str):
    print(f"#filter_min_interval_gap before gap:{min_interval_gap} length:{len(data)}")
    data = data.copy()

    last_valid_time = -np.inf  # Last timestamp where `below_min == True`
    is_below_min_streak = False  # Track whether we're in a streak of `below_min == True`

    for idx, row in data.iterrows():
        # Default to assigning "Filter Interval Gap"
        if pd.isna(data.loc[idx, 'filter']):
            data.loc[idx, 'filter'] = "Filter Interval Gap"

        if row['below_min']:
            is_below_min_streak = True  # We're in a streak of `below_min == True`
            last_valid_time = row['ts']  # Update last valid timestamp when `below_min` is True
        elif is_below_min_streak:
            # Transition from `below_min == True` to `below_min == False`
            gap = row['ts'] - last_valid_time  # Calculate the gap between current time and the last valid `True`
            if gap >= min_interval_gap * get_interval_coefficient(interval):
                # If the gap exceeds `min_interval_gap`, stop filtering at this point
                data.loc[idx, 'filter'] = None
            # Reset streak
            is_below_min_streak = False
    
    print(f"#filter_min_interval_gap after gap:{min_interval_gap} length:{len(data)}")
    return data

# def filter_min_interval_gap(data, min_interval_gap: int, interval: str):
#     print(f"#filter_min_interval_gap before gap:{min_interval_gap} length:{len(data)}")
#     data = data.copy()

#     last_valid_time = -np.inf  # Last timestamp when `below_min == True`
#     strike_count = 0  # Track the number of strikes
#     is_below_min_streak = False  # Track if we're in a streak of `below_min == True`

#     for idx, row in data.iterrows():
#         # Default to assigning "Filter Interval Gap" only if `filter` is None
#         if pd.isna(data.loc[idx, 'filter']):
#             data.loc[idx, 'filter'] = "Filter Below Min Strikes"
        
#         if row['below_min']:
#             # We're in a streak of `below_min == True`
#             is_below_min_streak = True
#             strike_count += 1
#             last_valid_time = row['ts']  # Update last valid timestamp when `below_min` is True
#         elif is_below_min_streak:
#             # We transition from `below_min == True` to `below_min == False`
#             gap = row['ts'] - last_valid_time  # Calculate the gap between current time and last valid `True`
            
#             if gap >= min_interval_gap * get_interval_coefficient(interval):
#                 # If the gap exceeds `min_interval_gap`, mark the current timestamp and all intermediate gaps as None
#                 # Set `filter` to None for this and all subsequent rows in the gap
#                 for i in range(idx - strike_count, idx):
#                     data.loc[i, 'filter'] = None
#                 strike_count = 0  # Reset strike count
#             # Reset streak
#             is_below_min_streak = False

#     print(f"#filter_min_interval_gap after gap:{min_interval_gap} length:{len(data)}")
#     return data

# def filter_below_min_strikes(all_data, filtered_data):
#     print(f"#filter_below_min_strikes before; length:{len(all_data)}")
#     all_data = all_data.copy()

#     is_below_min_streak = False  # Track whether we're in a streak of `below_min == True`

#     for idx, row in all_data.iterrows():
#         ts = row['ts']
#         if pd.isna(all_data.loc[idx, 'filter']):
#             all_data.loc[idx, 'filter'] = "Filter Below Min Strikes"
#             # Also update 'filtered_data' with the same timestamp
#             filtered_data.loc[filtered_data['ts'] == ts, 'filter'] = "Filter Below Min Strikes"

#         if row['below_min']:
#             is_below_min_streak = True  # We're in a streak
#         elif is_below_min_streak:
#             # Transition from `below_min == True` to `below_min == False`
#             is_below_min_streak = False  # End the streak
#             # Reverse logic: remove the filter only at transition points
#             all_data.loc[idx, 'filter'] = None
#             filtered_data.loc[filtered_data['ts'] == ts, 'filter'] = None  # Also update filtered_data
    
#     filtered_data = all_data[all_data['filter'].isna()] 
#     print(f"#filter_below_min_strikes after; length:{len(filtered_data)}")
#     return all_data, filtered_data

def long_signal_min_bsp(all_data):
    all_data.loc[all_data['below_min'], 'long_signal'] = True
    all_data.loc[~all_data['below_min'], 'long_signal'] = False


    print(f"#long_signal_min_bsp length:{len(all_data['long_signal'])}")
    return all_data

def long_signal_below_min_strikes(all_data):
    all_data = all_data.copy()

    # Track the streak of below_min == True
    is_below_min_streak = False

    # Store the timestamps (ts) where the filter should be applied
    filter_update_ts = []

    for idx, row in all_data.iterrows():
        ts = row['ts']

        # If below_min is True, mark the filter for all subsequent rows in the streak
        if row['below_min']:
            is_below_min_streak = True
        elif is_below_min_streak:
            # When below_min changes to False, mark the transition point and break the streak
            is_below_min_streak = False
            filter_update_ts.append(ts)  # Mark the transition point for filter reset

    # filter_update_ts = [np.int64(x) for x in filter_update_ts]
    # Apply "Filter Below Min Strikes" where needed by matching 'ts'
    # all_data.loc[all_data['ts'].isin(filter_update_ts), 'filter'] = None
    all_data.loc[all_data['ts'].isin(filter_update_ts), 'long_signal'] = True
    all_data.loc[~all_data['ts'].isin(filter_update_ts), 'long_signal'] = False

    print(f"#long_signal_below_min_strikes length:{len(all_data['long_signal'])}")
    return all_data

def produce_default_statistic(trades: pd.DataFrame, verbose=False):
    tsl_winning_trades = ((trades['tsl_percentage_result'] > 0) & (trades['status'] == "Trade")).sum()
    tsl_losing_trades = ((trades['tsl_percentage_result'] < 0) & (trades['status'] == "Trade")).sum()
    tsl_total_trades = tsl_winning_trades + tsl_losing_trades

    all_tsl_winning_trades = (trades['tsl_percentage_result'] > 0).sum()
    all_tsl_losing_trades = (trades['tsl_percentage_result'] < 0).sum()
    all_tsl_total_trades = all_tsl_winning_trades + all_tsl_losing_trades

    # Compute win ratios
    win_ratios = {
        'tsl': round(tsl_winning_trades / tsl_total_trades, 4) if tsl_total_trades > 0 else 0.0,
        'all_tsl': round(all_tsl_winning_trades / all_tsl_total_trades, 4) if all_tsl_total_trades > 0 else 0.0,
    }

    # Compute basic statistics
    stats = {
        # only winning
        'traded_winning_tsl': trades.loc[(trades['tsl_percentage_result'] > 0) & (trades['status'] == "Trade"), 'tsl_percentage_result'].describe() if tsl_winning_trades > 0 else "No winning TSL trades",
        'all_winning_tsl': trades.loc[trades['tsl_percentage_result'] > 0, 'tsl_percentage_result'].describe() if all_tsl_winning_trades > 0 else "No winning TSL",
        # all
        'traded_tsl': trades.loc[(trades['tsl_percentage_result'] != 0) & (trades['status'] == "Trade"), 'tsl_percentage_result'].describe() if tsl_total_trades > 0 else "No TSL trades",
        'all_tsl': trades.loc[trades['tsl_percentage_result'] != 0, 'tsl_percentage_result'].describe() if all_tsl_total_trades > 0 else "No TSL",
    }

    
    
    if verbose:
        print_win_ratios(win_ratios)
        print("\nStatistics for only winning 1) trades 2) trades + potential filtered trades:")
        print('traded_winning_tsl')
        print(stats['traded_winning_tsl'])
        print()
        print('all_winning_tsl')
        print(stats['all_winning_tsl'])
        print()
        print("Statistics for all (winning + lost) 1) trades 2) trades + potential filtered trades:")
        print('traded_tsl')
        print(stats['traded_tsl'])
        print()
        print('all_tsl')
        print(stats['all_tsl'])
        print()
        
    # Handle statistics that might be strings (i.e., "No winning TSL trades" or similar)
    statistic = {
        'winning_tsl': {} if isinstance(stats['traded_winning_tsl'], str) else stats['traded_winning_tsl'].to_dict(),
        'all_winning_tsl': {} if isinstance(stats['all_winning_tsl'], str) else stats['all_winning_tsl'].to_dict(),
        'traded_tsl': {} if isinstance(stats['traded_tsl'], str) else stats['traded_tsl'].to_dict(),
        'all_tsl': {} if isinstance(stats['all_tsl'], str) else stats['all_tsl'].to_dict(),
    }

    return win_ratios, statistic

def print_win_ratios(win_ratios):
    print("Win Ratios:")
    for key, value in win_ratios.items():
        print(f"{key}: {value:.4f}")