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
