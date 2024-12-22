from itertools import product
import time
import json
import matplotlib.pyplot as plt
from typing import TypedDict, Optional, Union
import pandas as pd
from .helpers import get_interval_coefficient, calculate_percentage_change, filter_min_interval_gap, filter_max_bsp, long_signal_below_min_strikes, filter_min_bsp, produce_default_statistic, long_signal_min_bsp, filter_bsp_required_min_value
from .strategies import calculate_trailing_exit

def compute_trades(
    interval: str,
    points_below_min: pd.DataFrame,
    data_clean: pd.DataFrame,
    past_interval_percentage: int = 0,
    past_percentage_min_dropdown: float = 0,
    tsl_trailing_stop_loss: float = 3,
    tsl_stop_loss: float = 5,
    tsl_take_profit: float = 0,
    use_avg_price: bool = False,
    show_chart: bool = True,
    points_above_max: Optional[pd.DataFrame] = None,
    chart_title: str = 'Cumulative Strategy Results',
    trade_options = {}
    ):
    transactions = []
    cumulative_in_trade_tsl_percentage_result = 0
    cumulative_all_tsl_percentage_result = 0

    data_after_entry = data_clean[['ts', 'h', 'l', 'c']]

    for index, point in points_below_min.iterrows():
        # if index % 500 == 0: # type: ignore
        #     print(f"Transaction: {point['ts']}, {index}")
        ts = point['ts']
        status = None
        price_change_percent = None
        tsl_exit_price = None
        tsl_percentage_result = None
        tsl_exit_reason = None

        # Determine entry price
        entry_price = (point['l'] + point['h']) / 2 if use_avg_price else point['c']

        if pd.notna(point['long_signal_filter']):
            status = point['long_signal_filter']
        # Check optional condition for past percentage drop
        elif past_interval_percentage != 0 and past_percentage_min_dropdown != 0:
            past_ts = ts - past_interval_percentage * get_interval_coefficient(interval)
            past_prices = data_clean.loc[(data_clean['ts'] >= past_ts) & (data_clean['ts'] < ts), 'h']
            max_past_price = past_prices.max() if not past_prices.empty else None # type: ignore

            if max_past_price is not None:
                drop_percent = calculate_percentage_change(entry_price, max_past_price)
                if drop_percent > past_percentage_min_dropdown:
                    status = "Price drop not enough"

        # Calculate price change percentage
        past_ts = ts - past_interval_percentage * get_interval_coefficient(interval)
        past_price = data_clean.loc[(data_clean['ts'] - past_ts).abs().idxmin(), 'l']
        price_change_percent = calculate_percentage_change(entry_price, past_price)
        
        # Get prices after entry
        data_after_entry = data_after_entry.loc[data_clean['ts'] >= ts]
        
        # Trailing stop-loss exit
        trail_exit = calculate_trailing_exit(point, data_after_entry, tsl_trailing_stop_loss, tsl_stop_loss, tsl_take_profit, points_above_max)
        tsl_exit_price = trail_exit['price'] if 'price' in trail_exit else None
        tsl_exit_reason = trail_exit['exit_reason']
        tsl_percentage_result = calculate_percentage_change(tsl_exit_price, entry_price) if tsl_exit_price else -tsl_stop_loss
        
        # Update cumulative results
        cumulative_in_trade_tsl_percentage_result += tsl_percentage_result if status is None else 0
        cumulative_all_tsl_percentage_result += tsl_percentage_result

        # Add transaction details
        transactions.append({
            'ts': ts,
            'entry_price': entry_price,
            'price_change_percent': price_change_percent,
            'tsl_exit_ts': trail_exit['ts'],
            'tsl_exit_price': tsl_exit_price,
            'tsl_percentage_result': tsl_percentage_result,
            'tsl_exit_reason': tsl_exit_reason,
            'cumulative_in_trade_tsl_percentage_result': cumulative_in_trade_tsl_percentage_result,
            'cumulative_all_tsl_percentage_result': cumulative_all_tsl_percentage_result,
            'status': status or "Trade"
        })

    trades_df = pd.DataFrame(transactions)
    tsl_percentage_result_traded = trades_df['cumulative_in_trade_tsl_percentage_result'].iloc[-1]
    tsl_percentage_result_traded_max = trades_df['cumulative_in_trade_tsl_percentage_result'].max()
    tsl_percentage_result_traded_min = trades_df['cumulative_in_trade_tsl_percentage_result'].min()

    tsl_percentage_result_all = trades_df['cumulative_all_tsl_percentage_result'].iloc[-1]
    tsl_percentage_result_all_max = trades_df['cumulative_all_tsl_percentage_result'].max()
    tsl_percentage_result_all_min = trades_df['cumulative_all_tsl_percentage_result'].min()
    
    # Calculate trade counts
    tsl_trade_count_closed = trades_df[(trades_df['tsl_exit_reason'].notnull()) & (trades_df['status'] == "Trade")].shape[0]
    tsl_trade_count_active = trades_df[(trades_df['tsl_exit_reason'].isnull()) & (trades_df['status'] == "Trade")].shape[0]
    
    # For 'all' counts, you can simply use the total number of rows
    all_tsl_count_closed = trades_df[trades_df['tsl_exit_reason'].notnull()].shape[0]
    all_tsl_count_active = trades_df[trades_df['tsl_exit_reason'].isnull()].shape[0]
    
    trades_stats={
        "tsl_percentage_result_traded":tsl_percentage_result_traded,
        "tsl_percentage_result_all":tsl_percentage_result_all,
        "tsl_trade_count_closed":tsl_trade_count_closed,
        "tsl_trade_count_active":tsl_trade_count_active,
        "all_tsl_count":all_tsl_count_closed,
        "all_tsl_count_active":all_tsl_count_active,
    }   

    if show_chart:
        # Calculate the HODL percentage change series
        initial_price = data_clean['c'].iloc[0]  # Assuming 'c' (close price) is used as the reference
        data_clean['hodl_percentage_change'] = ((data_clean['c'] - initial_price) / initial_price) * 100

        # Create the figure
        plt.figure(figsize=(12, 6))

        # Plot the different strategies
        plt.plot(trades_df['ts'], trades_df['cumulative_in_trade_tsl_percentage_result'], label='TSL Strategy Traded', color='red')
        plt.plot(trades_df['ts'], trades_df['cumulative_all_tsl_percentage_result'], label='TSL Strategy All', color='red', alpha=0.3)

         # Plot the price series for 'low' and 'high'
        # plt.plot(data_clean['ts'], data_clean['l'], label='Price Low', color='red', alpha=0.7)
        # plt.plot(data_clean['ts'], data_clean['h'], label='Price High', color='green', alpha=0.7)
        plt.plot(data_clean['ts'], data_clean['hodl_percentage_change'], label='HODL Percentage Change', color='green', linestyle='--', linewidth=1)

        # Annotate the last points with text
        plt.figtext(0.5, -0.05, f'TSL Traded: {tsl_percentage_result_traded:.2f}% ; closed: {tsl_trade_count_closed}; active: {tsl_trade_count_active}; min: {tsl_percentage_result_traded_min}; max: {tsl_percentage_result_traded_max}', ha='center', va='top', fontsize=10, color='red')
        plt.figtext(0.5, -0.1, f'TSL All: {tsl_percentage_result_all:.2f}% ; closed: {all_tsl_count_closed}; active: {all_tsl_count_active}; min: {tsl_percentage_result_all_min}; max: {tsl_percentage_result_all_max}', ha='center', va='top', fontsize=10, color='red', alpha=0.5)
        plt.figtext(0.5, -0.15, f'HODL: {data_clean["hodl_percentage_change"].iloc[-1]:.2f}%', ha='center', va='top', fontsize=10, color='green')

        # trade options info
        # options_text = ", ".join(f"{key}={value}" for key, value in trade_options.items())
        # chart_title = f"{chart_title} ({options_text})" if chart_title else options_text
        
        options_text = "\n".join(f"{key}: {value}" for key, value in trade_options.items())
        plt.text(
            0.01, 0.98, options_text,
            transform=plt.gca().transAxes,
            fontsize=7,
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')  # Background styling
        )

        # Labels, title, and grid
        plt.xlabel('Timestamp')
        plt.ylabel('Cumulative Result (%)')
        plt.title(chart_title)
        plt.legend()
        plt.grid()

        # Show the plot
        plt.show()

    return trades_df, trades_stats

class TradeOptions(TypedDict, total=False):
    bsp_back_interval_amount: int
    bsp_required_min_value: Optional[float]
    tsl_trailing_stop_loss: float
    tsl_stop_loss: float
    tsl_take_profit: float
    past_interval_percentage: Optional[int]
    past_percentage_min_dropdown: Optional[float]
    use_avg_price: Optional[bool]
    show_chart: Optional[bool]
    use_points_above_max: Optional[bool]
    chart_title: Optional[str]

def trade_simulation(interval: str, ticker: str, long_signal_generator_name: str, trade_options: TradeOptions, cut_potential_trades=None,
                     file_name=None, print_trades=False, filter_min_interval_gap_to_skip: int|None=None, show_statistic=False):
    
    bsp_back_interval_amount = trade_options.get('bsp_back_interval_amount') or 0
    bsp_required_min_value = trade_options.get('bsp_required_min_value') or 0
    past_interval_percentage = trade_options.get('past_interval_percentage') or 0
    past_percentage_min_dropdown = trade_options.get('past_percentage_min_dropdown')  or 0
    tsl_trailing_stop_loss = trade_options.get('tsl_trailing_stop_loss') or 3
    tsl_stop_loss = trade_options.get('tsl_stop_loss') or 5
    tsl_take_profit = trade_options.get('tsl_take_profit') or 0
    use_avg_price = trade_options.get('use_avg_price') or False
    show_chart = trade_options.get('show_chart') or False
    use_points_above_max = trade_options.get('use_points_above_max') or False 
    chart_title = trade_options.get('chart_title') or ''
    
    data = pd.read_json(f"../data/data-crypto-{ticker}-{interval}.json")
    
    # Handle missing values
    if data.isna().any().any():
        data_clean: pd.DataFrame = data.dropna().copy()  # Use .copy() to create an explicit copy
    else:
        data_clean: pd.DataFrame = data.copy()
    print(f"Number of records after clean: {len(data_clean)}")

    with pd.option_context('mode.chained_assignment', None): # turn off warning
        data_clean.loc[:, 'long_signal_filter'] = None
        data_clean.loc[:, 'long_signal'] = False
    
    data_clean = filter_min_bsp(data_clean, bsp_back_interval_amount)[0]
    data_clean, points_above_max= filter_max_bsp(data_clean, bsp_back_interval_amount)

    # should choose only one long entry option
    if long_signal_generator_name == 'long_signal_min_bsp':
        data_clean = long_signal_min_bsp(data_clean)
    elif long_signal_generator_name == 'long_signal_below_min_strikes':
        data_clean = long_signal_below_min_strikes(data_clean)

    # filters
    if bsp_required_min_value:
        data_clean = filter_bsp_required_min_value(data_clean, bsp_required_min_value)
    
    start_time = time.time()
    
    long_entries = data_clean[data_clean['long_signal']]
    print(f"Number of long entries: {len(long_entries)}")
    
    if cut_potential_trades is not None:
        long_entries=long_entries.loc[:cut_potential_trades]
	
    points_above_max=points_above_max if use_points_above_max else None
    trades, trades_stats = compute_trades(interval, long_entries, data_clean, past_interval_percentage, past_percentage_min_dropdown,
                                                  tsl_trailing_stop_loss, tsl_stop_loss, tsl_take_profit,
                                                  use_avg_price, show_chart, points_above_max, chart_title, trade_options)

    win_ratios, statistic = produce_default_statistic(trades, verbose=show_statistic)
    details={}
    details['win_ratios'] = win_ratios
    details['statstatistic'] = statistic
    details['default_stat'] = {
        "possible_long_entries": len(long_entries),
        **trades_stats
    }
    
    if print_trades:
        print(trades)
    elapsed_time = time.time() - start_time
    print(f"Time taken for transactions: {elapsed_time:.2f} seconds")

    if file_name:
        export_data = {
            "details": details,
            "trades": trades.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
        }
        with open(f"{file_name}.json", "w") as json_file:
            json.dump(export_data, json_file, indent=4)

    return {
        "win_ratios": win_ratios,
        "statistic": statistic,
    }
            
class TradeOptionsUnion(TypedDict, total=False):
    bsp_back_interval_amount: Union[int, list[int]]
    bsp_required_min_value: Union[float, list[float]]
    past_interval_percentage: Union[float, list[float]]
    past_percentage_min_dropdown: Union[float, list[float]]
    tsl_trailing_stop_loss: Union[float, list[float]]
    tsl_stop_loss: Union[float, list[float]]
    tsl_take_profit: Union[float, list[float]]
    use_avg_price: bool
    show_chart: bool
    use_points_above_max: bool
    chart_title: Union[str, None]

def trade_simulation_test_parameters(interval: str, ticker: str, long_signal_generator_name: str, 
                         trade_options: TradeOptionsUnion, cut_potential_trades=None, 
                         file_name=None, show_statistic=False):
    # Define keys that can accept arrays
    parameter_keys = [
        "bsp_back_interval_amount",
        "bsp_required_min_value",
        "past_interval_percentage",
        "past_percentage_min_dropdown",
        "tsl_trailing_stop_loss",
        "tsl_stop_loss",
        "tsl_take_profit"
    ]
    
    # Extract lists or single values from trade_options
    parameter_values = {key: (trade_options[key] if isinstance(trade_options[key], list) else [trade_options[key]])
                        for key in parameter_keys if key in trade_options}
    
    # Generate all combinations of the parameter values
    param_combinations = list(product(*parameter_values.values()))
    param_names = list(parameter_values.keys())  # To map combination values back to keys
    
    best_tsl_ratio = -float('inf')
    best_all_tsl_ratio = -float('inf')
    best_configuration = {}

    # Iterate over all parameter combinations
    for combination in param_combinations:
        # Create a new trade_options dict for this combination
        current_trade_options = trade_options.copy()
        current_trade_options.update(dict(zip(param_names, combination))) # type: ignore

        print(f"Testing combination: {current_trade_options}")
        
        # Call the trade_simulation function
        details = trade_simulation(
            interval=interval,
            ticker=ticker,
            long_signal_generator_name=long_signal_generator_name,
            trade_options=current_trade_options, # type: ignore
            cut_potential_trades=cut_potential_trades,
            file_name=file_name,
            show_statistic=show_statistic
        )
        
        # Retrieve the win_ratios from the details
        win_ratios = details['win_ratios'] # type: ignore
        tsl_ratio = win_ratios.get('tsl', 0)
        all_tsl_ratio = win_ratios.get('all_tsl', 0)
        
        # Update the best configuration based on tsl ratio
        if tsl_ratio > best_tsl_ratio:
            best_tsl_ratio = tsl_ratio
            best_configuration['tsl'] = {
                'trade_options': current_trade_options,
                'win_ratio': tsl_ratio
            }
        
        # Update the best configuration based on all_tsl ratio
        if all_tsl_ratio > best_all_tsl_ratio:
            best_all_tsl_ratio = all_tsl_ratio
            best_configuration['all_tsl'] = {
                'trade_options': current_trade_options,
                'win_ratio': all_tsl_ratio
            }

    print("\nBest Configuration Based on TSL Win Ratio:")
    print(best_configuration['tsl'])
    print("\nBest Configuration Based on All TSL Win Ratio:")
    print(best_configuration['all_tsl'])
    
    return best_configuration