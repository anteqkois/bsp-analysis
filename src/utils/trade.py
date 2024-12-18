import time
import matplotlib.pyplot as plt
from typing import TypedDict, Optional
import pandas as pd
from helpers import get_interval_coefficient, calculate_percentage_change, filter_min_interval_gap, addMinMaxBSP
from strategies import calculate_trailing_exit, calculate_fixed_exit

def compute_trades(
    interval: str,
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

class TradeOptions(TypedDict):
    past_interval_percentage: int
    past_percentage_min_dropdown: float
    tsl_trailing_stop_loss: float
    tsl_stop_loss: float
    tsl_take_profit: float
    fixed_take_profit: float
    fixed_stop_loss: float
    use_avg_price: Optional[bool]
    show_chart: Optional[bool]
    use_points_above_max: Optional[bool]
    chart_title: Optional[str]

def trade_simulation(interval: str, ticker: str, filter_min_interval_gap_to_skip: int, back_interval_amount_for_bsp: int, trade_options: TradeOptions):
    data = pd.read_json(f"../data/data-crypto-{ticker}-{interval}.json")
    
    # Handle missing values
    if data.isna().any().any():
        data_clean: pd.DataFrame = data.dropna().copy()  # Use .copy() to create an explicit copy
    else:
        data_clean: pd.DataFrame = data.copy()
    print(f"Number of records after clean: {len(data_clean)}")
    
    data_clean, points_below_min, points_above_max= addMinMaxBSP(data_clean, back_interval_amount_for_bsp)
    
    points_below_min = filter_min_interval_gap(points_below_min, filter_min_interval_gap_to_skip, interval)
    
    start_time = time.time()

    past_interval_percentage = trade_options.get('past_interval_percentage')
    past_percentage_min_dropdown = trade_options.get('past_percentage_min_dropdown') 
    tsl_trailing_stop_loss = trade_options.get('tsl_trailing_stop_loss') 
    tsl_stop_loss = trade_options.get('tsl_stop_loss') 
    tsl_take_profit = trade_options.get('tsl_take_profit') 
    fixed_take_profit = trade_options.get('fixed_take_profit') 
    fixed_stop_loss = trade_options.get('fixed_stop_loss') 
    use_avg_price = trade_options.get('use_avg_price') or False
    show_chart = trade_options.get('show_chart') or False
    use_points_above_max = trade_options.get('use_points_above_max') or False 
    chart_title = trade_options.get('chart_title') or ''
    points_above_max=points_above_max if use_points_above_max else None
	
    transactions_without_bsp_max = compute_trades(interval, points_below_min, data_clean, past_interval_percentage, past_percentage_min_dropdown,
                                                  tsl_trailing_stop_loss, tsl_stop_loss, tsl_take_profit, fixed_take_profit, fixed_stop_loss,
                                                  use_avg_price, show_chart, points_above_max, chart_title)
    elapsed_time = time.time() - start_time
    print(f"Time taken for transactions: {elapsed_time:.2f} seconds")