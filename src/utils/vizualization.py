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