# %%
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install numpy pandas scipy matplotlib # type: ignore
# import json
from utils.trade import trade_simulation
 

# %%
trade_simulation(
    interval='1m',
    ticker='TAOUSDT',
    back_interval_amount_for_bsp=1000,
    filter_min_interval_gap_to_skip=2,
    # cut_potential_trades=1000,
    file_name='base',
    trade_options={
        "past_interval_percentage": 120,
        "past_percentage_min_dropdown": -2.5,
        "tsl_trailing_stop_loss": 5,
        "tsl_stop_loss": 10,
        "tsl_take_profit": 6,
        "fixed_take_profit": 6,
        "fixed_stop_loss": 5,
        "use_avg_price": False,
        "show_chart": True,
        "use_points_above_max": False,
        "chart_title": None
    }
)

# # %%
# # # Trades without BSP max
# transactions_without_bsp_max = compute_trades(interval, points_below_min, data_clean, past_interval_percentage=120, past_percentage_min_dropdown=-2.5,
#                                  tsl_trailing_stop_loss=5, tsl_stop_loss=10, tsl_take_profit=6, fixed_take_profit=6, fixed_stop_loss=5)

# # %%
# # Trades with BSP max
# transactions_with_bsp_max = compute_trades(interval, points_below_min, data_clean, past_interval_percentage=120, past_percentage_min_dropdown=-2.5,
#                                  tsl_trailing_stop_loss=5, tsl_stop_loss=10, tsl_take_profit=6, fixed_take_profit=6, fixed_stop_loss=5,
#                                 points_above_max=points_above_max, chart_title='SL=False TSL=False TP=False BSP=True')

# # %%
# # Trades with BSP max
# transactions_with_bsp_max = compute_trades(interval, points_below_min, data_clean, past_interval_percentage=120, past_percentage_min_dropdown=-2.5,
#                                  tsl_trailing_stop_loss=5, tsl_stop_loss=10, tsl_take_profit=8, fixed_take_profit=8, fixed_stop_loss=5,
#                                 points_above_max=points_above_max, chart_title='SL=True TSL=True TP=True BSP=True')
# # %%

# %%
