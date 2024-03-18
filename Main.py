import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import datetime
import mplcyberpunk
import matplotlib.pyplot as plt


# Suppress specific FutureWarnings to clean up output
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define stock ticker, start and end date for data retrieval
ticker = "TWLO"
start = "2020-01-01"
end = datetime.datetime.now()

# Download stock data from Yahoo Finance
data = yf.download(ticker, start, end)

# Calculate daily return and determine if the day's return is up or down
data["Daily_Return"] = data["Adj Close"].pct_change()
data["State"] = np.where(data["Daily_Return"] >= 0, "up", "down")

# Display the stock ticker and the data frame with the new columns
print('Daily Stock Name = ', ticker)
#print(data)
print(data[['Adj Close', 'Daily_Return', 'State']].tail())  # Show only the last few rows for brevity

# Calculate and display the counts of up and down days
counts = data["State"].value_counts(normalize=True) * 100
print("Percentage of 'UP' and 'DOWN' days:\n", counts)

# Calculate the transition matrix for state changes
transition_matrix = pd.crosstab(data["State"], data["State"].shift(-1), normalize='index') * 100
transition_matrix.columns = ["To Down", "To Up"]
transition_matrix.index = ["From Down", "From Up"]
print("\nTransition Matrix (%):\n", transition_matrix)

def consecutive_state_probability(data, state="down", following_state="up", consecutive_days=6):
    """
    Calculates the probability of a following_state day after a sequence of consecutive state days.
    """
    condition = (data["State"] == state)
    for i in range(1, consecutive_days):
        condition &= (data["State"].shift(i) == state)
    following_condition = condition & (data["State"].shift(-1) == following_state)

    if condition.sum() > 0:  # Check if there are any instances meeting the condition
        probability = (following_condition.sum() / condition.sum()) * 100
        return probability
    else:
        return np.nan  # or return 0, or return "N/A", depending on how you want to handle this case

# Example: Calculate and display probabilities for various scenarios
consecutive_days_to_test = [5, 6, 7, 8, 9, 10, 11]
for days in consecutive_days_to_test:
    probability = consecutive_state_probability(data, state="down", following_state="up", consecutive_days=days)
    if np.isnan(probability):
        print(f"Probability (%) of going up after exactly {days} consecutive down days: N/A (no occurrences found)")
    else:
        print(f"Probability (%) of going up after exactly {days} consecutive down days: {probability:.2f}%")

# Set display options for pandas DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4096)

def is_local_extremum(df_close, curr_index, order, extremum_type='top'):
    """
    Checks if a point is a local top or bottom.
    
    :param df_close: Numpy array of close prices
    :param curr_index: Current index in df_close
    :param order: Number of points to look around the current point
    :param extremum_type: Type of extremum ('top' or 'bottom')
    :return: Boolean indicating if the point is a local extremum
    """
    if curr_index < order * 2 + 1:
        return False

    k = curr_index - order
    v = df_close[k]
    if extremum_type == 'top':
        return all(v > df_close[k - i] and v > df_close[k + i] for i in range(1, order + 1))
    else:  # extremum_type == 'bottom'
        return all(v < df_close[k - i] and v < df_close[k + i] for i in range(1, order + 1))

def local_extrema(df, order, threshold, only_extrema):
    """
    Finds local tops and bottoms in a DataFrame.

    :param df: DataFrame with stock data
    :param order: Number of points to look around the current point for extrema
    :param threshold: Threshold for confirming trend reversals
    :param only_extrema: Flag to include only extrema or also confirmatory points
    :return: Lists of tops and bottoms
    """
    tops = []
    bottoms = []
    up_trend = False
    confirm_i = 0

    df_close = df['close'].to_numpy()
    tmp_min = df.at[df.index[0], 'low']
    tmp_max = df.at[df.index[0], 'high']
    tmp_min_i = 0
    tmp_max_i = 0

    for i in range(len(df)):
        now_high = df.at[df.index[i], 'high']
        now_low = df.at[df.index[i], 'low']
        now_close = df.at[df.index[i], 'close']
        
        if up_trend:
            if now_close > tmp_max:
                tmp_max = now_close
                tmp_max_i = i

            if i - order > confirm_i and (is_local_extremum(df_close, i, order, 'top') or 
                                          (not only_extrema and now_close < tmp_max - tmp_max * threshold)):
                tops.append([i, df_close[i] * 1.1, i - order, df_close[i - order]])
                up_trend = False
                confirm_i = i
                tmp_min = now_low
                tmp_min_i = i

        else:
            if now_close < tmp_min:
                tmp_min = now_close
                tmp_min_i = i

            if i - order > confirm_i and (is_local_extremum(df_close, i, order, 'bottom') or 
                                          (not only_extrema and now_close > tmp_min + tmp_min * threshold)):
                bottoms.append([i, df_close[i] * 0.9, i - order, df_close[i - order]])
                up_trend = True
                confirm_i = i
                tmp_max = now_high
                tmp_max_i = i

    return tops, bottoms

if __name__ == "__main__":
    start_date = start
    end_date = end

    order = 15
    threshold = 0.1

    show_extrema = False
    show_confirm = True

    only_extrema = True

    # Fetch data
    df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    df.columns = df.columns.str.lower()

    tops, bottoms = local_extrema(df, order, threshold, only_extrema)

    plt.style.use("cyberpunk")
    plt.figure(figsize=(10,6))
    df['close'].plot()

    mplcyberpunk.add_underglow()
    mplcyberpunk.add_glow_effects()
    mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)

    # Plotting logic remains the same
    for top in tops:
        if show_confirm: plt.plot(df.index[top[0]], top[1], marker='v', markersize=11, color='yellow')
        if show_extrema: plt.plot(df.index[top[2]], top[3], marker='v', markersize=11, color='gold')

    for bottom in bottoms:
        if show_confirm: plt.plot(df.index[bottom[0]], bottom[1], marker='^', markersize=11, color='white')
        if show_extrema: plt.plot(df.index[bottom[2]], bottom[3], marker='^', markersize=11, color='silver')

    plt.savefig('extrEMA_Analysis.png')
    plt.show()
    print('Daily Stock Name = ', ticker)
    print('extrEMA_Analysis.png Saved', end)
exit()