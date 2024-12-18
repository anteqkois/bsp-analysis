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

def addMinMaxBSP(data: pd.DataFrame, back_interval_amount: int):
    # Calculate min_bsp, and max_bsp
    data.loc[:, 'min_bsp'] = data['ob_10_p_l'].rolling(window=back_interval_amount, min_periods=1).min()
    data.loc[:, 'max_bsp'] = data['ob_10_p_h'].rolling(window=back_interval_amount, min_periods=1).max()

    # Detect points below min_bsp, and above max_bsp
    data.loc[:, 'below_min'] = data['ob_10_p_l'] <= data['min_bsp']
    data.loc[:, 'above_max'] = data['ob_10_p_h'] >= data['max_bsp']

    # Extract points below min_bsp and points above max
    points_below_min = data[data['below_min']]
    print(f"Number of points below min: {len(points_below_min)}")
    points_above_max = data[data['above_max']]
    print(f"Number of points above max: {len(points_above_max)}")
    
    return data, points_below_min, points_above_max

def filter_min_interval_gap(data, min_interval_gap: int, interval: str):
    data = data.copy()

    data.loc[:, 'intervalGapFilter'] = False
    last_valid_time = -np.inf

    for idx, row in data.iterrows():
        if row['below_min']:
            if (row['ts'] - last_valid_time) >= min_interval_gap * get_interval_coefficient(interval):
                data.at[idx, 'intervalGapFilter'] = True
                last_valid_time = row['ts']

    return data