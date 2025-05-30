import pandas as pd
import matplotlib.pyplot as plt # Re-enabled for plotting
import matplotlib.dates as mdates # Re-enabled for plotting

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
    Portfolio 1: QQQ_Return - IWM_Return (No leverage)
    Portfolio 2: SPY_Return - IWM_Return (No leverage)
    Portfolio 3: Long QQQ
    Portfolio 4: Long SPY
    Portfolio 5: Long IWM
    """
    portfolios = pd.DataFrame(index=merged_df.index)

    # Changed from 2x leverage to 1x (no leverage)
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
    daily_cumulative_growth = (1 + portfolio_returns).cumprod() * initial_investment

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

    extreme_days_results = {}
    for portfolio_name in portfolio_returns.columns: # Use original portfolio_returns for daily % change
        daily_returns_pct = portfolio_returns[portfolio_name] * 100
        best_days = daily_returns_pct.nlargest(7)
        worst_days = daily_returns_pct.nsmallest(7)
        extreme_days_results[portfolio_name] = {
            'best_days': best_days,
            'worst_days': worst_days
        }

        print(f"\n--- Extreme Performing Days for {portfolio_name} ---")
        print("Top 7 Best Days (%):")
        print(best_days.round(2))
        print("\nTop 7 Worst Days (%):")
        print(worst_days.round(2))

    return daily_cumulative_growth # Return the daily cumulative growth for plotting

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