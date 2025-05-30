import pandas as pd
import matplotlib.pyplot as plt # Re-enabled for plotting
import matplotlib.dates as mdates # Re-enabled for plotting
import numpy as np # Added for np.isfinite

def load_and_preprocess_data(file_path, asset_name):
    """
    Loads historical data from a CSV file, preprocesses it,
    and calculates daily returns.
    """
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        # Convert 'Change %' from string like '0.12%' to float 0.0012
        df['Daily Return'] = df['Change %'].astype(str).str.rstrip('%').astype('float') / 100.0
        df = df.set_index('Date')
        df = df[['Daily Return']]
        df.rename(columns={'Daily Return': f'{asset_name}_Return'}, inplace=True)
        return df.sort_index()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def calculate_portfolio_returns(merged_df):
    """
    Calculates daily returns for the five defined portfolios.
    Portfolio 1: QQQ_Return - IWM_Return (No leverage in daily return calculation)
    Portfolio 2: SPY_Return - IWM_Return (No leverage in daily return calculation)
    Portfolio 3: Long QQQ
    Portfolio 4: Long SPY
    Portfolio 5: Long IWM
    """
    portfolios = pd.DataFrame(index=merged_df.index)

    # Daily returns are calculated without leverage here
    portfolios['P1_QQQ_vs_IWM'] = merged_df['QQQ_Return'] - merged_df['IWM_Return']
    portfolios['P2_SPY_vs_IWM'] = merged_df['SPY_Return'] - merged_df['IWM_Return']
    portfolios['P3_Long_QQQ'] = merged_df['QQQ_Return']
    portfolios['P4_Long_SPY'] = merged_df['SPY_Return']
    portfolios['P5_Long_IWM'] = merged_df['IWM_Return']

    return portfolios

def analyze_portfolios(portfolio_returns, initial_investment=100):
    """
    Calculates daily and weekly growth, identifies best/worst performing days,
    saves daily cumulative growth to a CSV file, and returns daily cumulative growth.
    """
    # Calculate daily cumulative growth
    daily_cumulative_growth = pd.DataFrame(index=portfolio_returns.index)

    # Portfolio 1: 2x (QQQ - IWM) synthetic, non-compounded leverage on initial base
    if 'P1_QQQ_vs_IWM' in portfolio_returns.columns:
        # Daily unleveraged spread return
        daily_spread_P1 = portfolio_returns['P1_QQQ_vs_IWM']
        # Cumulative sum of these unleveraged spread returns
        cumulative_sum_spread_P1 = daily_spread_P1.cumsum()
        # Portfolio value = Initial_Base * (1 + 2 * Sum_of_Spread_Returns)
        daily_cumulative_growth['P1_QQQ_vs_IWM'] = initial_investment * (1 + 2 * cumulative_sum_spread_P1)
        # Max loss is initial investment; portfolio value cannot go below zero.
        daily_cumulative_growth['P1_QQQ_vs_IWM'] = daily_cumulative_growth['P1_QQQ_vs_IWM'].clip(lower=0)

    # Portfolio 2: 2x (SPY - IWM) synthetic, non-compounded leverage on initial base
    if 'P2_SPY_vs_IWM' in portfolio_returns.columns:
        # Daily unleveraged spread return
        daily_spread_P2 = portfolio_returns['P2_SPY_vs_IWM']
        # Cumulative sum of these unleveraged spread returns
        cumulative_sum_spread_P2 = daily_spread_P2.cumsum()
        # Portfolio value = Initial_Base * (1 + 2 * Sum_of_Spread_Returns)
        daily_cumulative_growth['P2_SPY_vs_IWM'] = initial_investment * (1 + 2 * cumulative_sum_spread_P2)
        # Max loss is initial investment; portfolio value cannot go below zero.
        daily_cumulative_growth['P2_SPY_vs_IWM'] = daily_cumulative_growth['P2_SPY_vs_IWM'].clip(lower=0)

    # Portfolio 3: Long QQQ (standard daily compounding)
    if 'P3_Long_QQQ' in portfolio_returns.columns:
        daily_cumulative_growth['P3_Long_QQQ'] = (1 + portfolio_returns['P3_Long_QQQ']).cumprod() * initial_investment

    # Portfolio 4: Long SPY (standard daily compounding)
    if 'P4_Long_SPY' in portfolio_returns.columns:
        daily_cumulative_growth['P4_Long_SPY'] = (1 + portfolio_returns['P4_Long_SPY']).cumprod() * initial_investment

    # Portfolio 5: Long IWM (standard daily compounding)
    if 'P5_Long_IWM' in portfolio_returns.columns:
        daily_cumulative_growth['P5_Long_IWM'] = (1 + portfolio_returns['P5_Long_IWM']).cumprod() * initial_investment

    # --- Create and save the CSV file for daily cumulative growth ---
    daily_cumulative_growth_for_csv = daily_cumulative_growth.copy()
    # Rename columns for clarity in CSV to indicate cumulative growth
    # This renaming is done on a copy, so original daily_cumulative_growth column names are preserved for plotting legend
    daily_cumulative_growth_for_csv.columns = [f'{col}_Cumulative_Growth' for col in daily_cumulative_growth_for_csv.columns]
    daily_cumulative_growth_for_csv.index.name = 'Date' # Set index name for CSV header

    csv_filename = 'daily_cumulative_portfolio_growth.csv'
    try:
        # Round to 2 decimal places for currency-like values in CSV
        daily_cumulative_growth_for_csv.round(2).to_csv(csv_filename)
        print(f"\nDaily cumulative growth of portfolios saved to {csv_filename}")
    except Exception as e:
        print(f"Error saving CSV file {csv_filename}: {e}")
    # --- End of CSV saving ---

    # Weekly growth table (for console output)
    weekly_growth_table_console = daily_cumulative_growth.resample('W').last()
    print("\n--- Weekly Growth of $100 Initial Investment (Console Display) ---")
    print(weekly_growth_table_console.ffill().round(2))

    # --- Best/Worst Performing Days and Worst Drawdown for each portfolio ---
    # Calculate actual daily percentage returns from the daily_cumulative_growth
    actual_daily_portfolio_returns = daily_cumulative_growth.pct_change()
    # For the first day, pct_change is NaN. Calculate it based on initial_investment.
    if not daily_cumulative_growth.empty and not actual_daily_portfolio_returns.empty:
        if len(daily_cumulative_growth) > 0 and len(actual_daily_portfolio_returns) > 0: # Check if there's at least one row
            first_day_values = daily_cumulative_growth.iloc[0]
            # Calculate first day returns, ensuring initial_investment is not zero
            if initial_investment != 0:
                first_day_returns = (first_day_values / initial_investment) - 1
            else: # Handle case where initial_investment is 0 (leads to NaNs or Infs)
                first_day_returns = pd.Series(np.nan, index=first_day_values.index)
            actual_daily_portfolio_returns.iloc[0] = first_day_returns

    print("\n\n--- Portfolio Performance Metrics ---")
    for portfolio_name in daily_cumulative_growth.columns: # Iterate over portfolios present in growth data
        print(f"\n--- Metrics for {portfolio_name} ---")

        # Best/Worst Days (based on actual daily % change of portfolio value)
        if portfolio_name in actual_daily_portfolio_returns.columns and actual_daily_portfolio_returns[portfolio_name].notna().any():
            daily_returns_pct = actual_daily_portfolio_returns[portfolio_name] * 100
            # Clean out NaN or Inf values
            daily_returns_pct_cleaned = daily_returns_pct.replace([np.inf, -np.inf], np.nan).dropna()

            if not daily_returns_pct_cleaned.empty:
                best_days = daily_returns_pct_cleaned.nlargest(7)
                worst_days = daily_returns_pct_cleaned.nsmallest(7)
                print("Top 7 Best Days (Actual Daily % Change of Portfolio Value):")
                print(best_days.round(2))
                print("\nTop 7 Worst Days (Actual Daily % Change of Portfolio Value):")
                print(worst_days.round(2))
            else:
                print("Best/Worst Days: Not available (no valid daily returns data after cleaning).")
        else:
            print("Best/Worst Days: Not available (no daily returns data for this portfolio).")

        # Worst Drawdown
        portfolio_values = daily_cumulative_growth[portfolio_name]
        # Ensure series is not empty and has some non-NaN values
        if not portfolio_values.empty and portfolio_values.notna().any():
            rolling_max = portfolio_values.cummax()
            drawdown_series = pd.Series(index=portfolio_values.index, dtype=float)

            for date_idx in portfolio_values.index:
                current_val = portfolio_values[date_idx]
                peak_val = rolling_max[date_idx]

                if pd.isna(current_val) or pd.isna(peak_val): # Handle NaNs in input
                    drawdown_series[date_idx] = np.nan
                elif peak_val == 0: # Implies current_val is also 0 (due to cummax and clip(lower=0))
                    drawdown_series[date_idx] = 0.0 # No drawdown if peak (and current) is 0
                else:
                    drawdown_series[date_idx] = (current_val / peak_val) - 1.0
            
            if drawdown_series.notna().any(): # Check if there's any valid drawdown calculated
                current_worst_drawdown = drawdown_series.min() * 100 # as percentage
                print(f"\nWorst Drawdown: {current_worst_drawdown:.2f}%")
            else:
                print("\nWorst Drawdown: Not available (drawdown series is all NaN).")
        else:
            print("\nWorst Drawdown: Not available (no portfolio value data).")

    return daily_cumulative_growth

def plot_daily_portfolio_growth(daily_cumulative_growth_df):
    """
    Plots the daily cumulative growth of all portfolios.
    """
    if daily_cumulative_growth_df.empty:
        print("No data available to plot.")
        return

    plt.figure(figsize=(15, 8)) # Adjusted figure size for better readability
    for column in daily_cumulative_growth_df.columns:
        plt.plot(daily_cumulative_growth_df.index, daily_cumulative_growth_df[column], label=column)

    plt.title(f'Daily Cumulative Growth of $100 Initial Investment (Starting {daily_cumulative_growth_df.index.min().strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Format the x-axis for dates
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12)) # Auto-adjust date ticks
    plt.xticks(rotation=45)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    # Save the plot
    plot_filename = 'daily_portfolio_cumulative_growth.png'
    try:
        plt.savefig(plot_filename)
        print(f"\nPlot saved as {plot_filename}")
    except Exception as e:
        print(f"Error saving plot file {plot_filename}: {e}")

    plt.show()


def main():
    # --- User-configurable start date ---
    USER_CONFIGURABLE_START_DATE = "2021-06-01"
    # --- End of User-configurable start date ---

    # Define file paths
    qqq_file = 'QQQ ETF Stock Price History.csv'
    spy_file = 'SPY ETF Stock Price History.csv'
    iwm_file = 'IWM ETF Stock Price History.csv'

    # Load data
    qqq_data = load_and_preprocess_data(qqq_file, 'QQQ')
    spy_data = load_and_preprocess_data(spy_file, 'SPY')
    iwm_data = load_and_preprocess_data(iwm_file, 'IWM')

    if qqq_data is None or spy_data is None or iwm_data is None:
        print("Exiting due to data loading errors. Please ensure CSV files are present and correctly named.")
        return

    # Merge dataframes on date index (inner join to get common dates)
    merged_df = pd.concat([qqq_data, spy_data, iwm_data], axis=1, join='inner')

    if merged_df.empty:
        print("No common dates found across the provided CSV files. Cannot proceed.")
        return

    common_start_date = merged_df.index.min()
    common_end_date = merged_df.index.max()
    print(f"Common data available from {common_start_date.strftime('%Y-%m-%d')} to {common_end_date.strftime('%Y-%m-%d')}")

    # Set analysis start date based on user configuration
    analysis_start_date = pd.to_datetime(USER_CONFIGURABLE_START_DATE)
    print(f"User-defined analysis start date: {analysis_start_date.strftime('%Y-%m-%d')}")

    # Filter data from the analysis_start_date onwards
    merged_df_filtered = merged_df[merged_df.index >= analysis_start_date]

    if merged_df_filtered.empty:
        print(f"No data available on or after the specified start date: {analysis_start_date.strftime('%Y-%m-%d')}. Please check CSV files and date ranges.")
        return

    actual_start_analysis = merged_df_filtered.index.min().strftime('%Y-%m-%d')
    actual_end_analysis = merged_df_filtered.index.max().strftime('%Y-%m-%d')
    print(f"Analyzing data from {actual_start_analysis} to {actual_end_analysis}")

    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(merged_df_filtered)

    if portfolio_returns.empty:
        print("Portfolio returns could not be calculated. Check data and date range.")
        return

    # Analyze portfolios (this will save CSV, print to console, and return daily growth)
    daily_cumulative_growth_df = analyze_portfolios(portfolio_returns)

    # Plot results if data is available
    if daily_cumulative_growth_df is not None and not daily_cumulative_growth_df.empty:
        plot_daily_portfolio_growth(daily_cumulative_growth_df)
    else:
        print("No daily cumulative growth data to plot.")

    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()