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
    analyzes max drawdown periods, saves results to CSV files,
    and returns daily cumulative growth.
    """
    # Calculate daily cumulative growth
    daily_cumulative_growth = pd.DataFrame(index=portfolio_returns.index)

    # Portfolio 1: 2x (QQQ - IWM) synthetic, non-compounded leverage on initial base
    if 'P1_QQQ_vs_IWM' in portfolio_returns.columns:
        daily_spread_P1 = portfolio_returns['P1_QQQ_vs_IWM']
        cumulative_sum_spread_P1 = daily_spread_P1.cumsum()
        daily_cumulative_growth['P1_QQQ_vs_IWM'] = initial_investment * (1 + 2 * cumulative_sum_spread_P1)
        daily_cumulative_growth['P1_QQQ_vs_IWM'] = daily_cumulative_growth['P1_QQQ_vs_IWM'].clip(lower=0)

    # Portfolio 2: 2x (SPY - IWM) synthetic, non-compounded leverage on initial base
    if 'P2_SPY_vs_IWM' in portfolio_returns.columns:
        daily_spread_P2 = portfolio_returns['P2_SPY_vs_IWM']
        cumulative_sum_spread_P2 = daily_spread_P2.cumsum()
        daily_cumulative_growth['P2_SPY_vs_IWM'] = initial_investment * (1 + 2 * cumulative_sum_spread_P2)
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
    daily_cumulative_growth_for_csv.columns = [f'{col}_Cumulative_Growth' for col in daily_cumulative_growth_for_csv.columns]
    daily_cumulative_growth_for_csv.index.name = 'Date'
    csv_filename_growth = 'daily_cumulative_portfolio_growth.csv'
    try:
        daily_cumulative_growth_for_csv.round(2).to_csv(csv_filename_growth)
        print(f"\nDaily cumulative growth of portfolios saved to {csv_filename_growth}")
    except Exception as e:
        print(f"Error saving CSV file {csv_filename_growth}: {e}")
    # --- End of CSV saving ---

    # Weekly growth table (for console output)
    weekly_growth_table_console = daily_cumulative_growth.resample('W').last()
    print("\n--- Weekly Growth of $100 Initial Investment (Console Display) ---")
    print(weekly_growth_table_console.ffill().round(2))

    # --- Best/Worst Performing Days and Max Drawdown Analysis ---
    actual_daily_portfolio_returns = daily_cumulative_growth.pct_change()
    if not daily_cumulative_growth.empty and not actual_daily_portfolio_returns.empty:
        if len(daily_cumulative_growth) > 0 and len(actual_daily_portfolio_returns) > 0:
            first_day_values = daily_cumulative_growth.iloc[0]
            if initial_investment != 0:
                first_day_returns = (first_day_values / initial_investment) - 1
            else:
                first_day_returns = pd.Series(np.nan, index=first_day_values.index)
            actual_daily_portfolio_returns.iloc[0] = first_day_returns

    all_extreme_days_data = []
    all_max_drawdown_data = []

    print("\n\n--- Portfolio Performance Metrics ---") # Keep console print for progress
    for portfolio_name in daily_cumulative_growth.columns:
        print(f"\n--- Analyzing Metrics for {portfolio_name} ---")

        # Best/Worst Days
        if portfolio_name in actual_daily_portfolio_returns.columns and actual_daily_portfolio_returns[portfolio_name].notna().any():
            daily_returns_pct = actual_daily_portfolio_returns[portfolio_name] * 100
            daily_returns_pct_cleaned = daily_returns_pct.replace([np.inf, -np.inf], np.nan).dropna()

            if not daily_returns_pct_cleaned.empty:
                best_days_series = daily_returns_pct_cleaned.nlargest(7)
                worst_days_series = daily_returns_pct_cleaned.nsmallest(7)

                for date, pct_change in best_days_series.items():
                    all_extreme_days_data.append({
                        'Portfolio': portfolio_name,
                        'Date': date.strftime('%Y-%m-%d'),
                        'Type': 'Best Day',
                        'Performance (%)': round(pct_change, 2)
                    })
                for date, pct_change in worst_days_series.items():
                    all_extreme_days_data.append({
                        'Portfolio': portfolio_name,
                        'Date': date.strftime('%Y-%m-%d'),
                        'Type': 'Worst Day',
                        'Performance (%)': round(pct_change, 2)
                    })
                # Console printing kept for verbosity during run, can be removed if not needed
                print("Top 7 Best Days (Actual Daily % Change of Portfolio Value):")
                print(best_days_series.round(2))
                print("\nTop 7 Worst Days (Actual Daily % Change of Portfolio Value):")
                print(worst_days_series.round(2))

            else:
                print(f"Best/Worst Days for {portfolio_name}: Not available (no valid daily returns data after cleaning).")
        else:
            print(f"Best/Worst Days for {portfolio_name}: Not available (no daily returns data).")

        # Max Drawdown Analysis
        portfolio_values = daily_cumulative_growth[portfolio_name]
        if not portfolio_values.empty and portfolio_values.notna().any():
            rolling_max = portfolio_values.cummax()
            drawdown_series_val = pd.Series(index=portfolio_values.index, dtype=float)
            peak_dates = pd.Series(index=portfolio_values.index, dtype='datetime64[ns]')

            current_peak_val = -np.inf
            current_peak_date = pd.NaT

            for date_idx in portfolio_values.index:
                if portfolio_values[date_idx] > current_peak_val : # Handles NaNs correctly, pd.NA > anything is false
                    current_peak_val = portfolio_values[date_idx]
                    current_peak_date = date_idx
                
                peak_dates[date_idx] = current_peak_date # Date of the current peak for this day
                
                # Calculate drawdown for the day
                peak_val_for_day = rolling_max[date_idx] # This is the true peak up to date_idx
                current_val = portfolio_values[date_idx]

                if pd.isna(current_val) or pd.isna(peak_val_for_day):
                    drawdown_series_val[date_idx] = np.nan
                elif peak_val_for_day == 0:
                    drawdown_series_val[date_idx] = 0.0
                else:
                    drawdown_series_val[date_idx] = (current_val / peak_val_for_day) - 1.0
            
            if drawdown_series_val.notna().any():
                worst_drawdown_pct = drawdown_series_val.min()
                trough_date = drawdown_series_val.idxmin()
                
                # Find the peak date corresponding to this trough
                # The peak for the trough_date is rolling_max[trough_date]
                # We need to find when this peak value first occurred.
                peak_value_at_trough_peak = rolling_max[trough_date]
                
                # Find the first date this peak_value_at_trough_peak was achieved before or at trough_date
                # Filter portfolio_values up to trough_date where value was the peak
                relevant_peaks = portfolio_values[portfolio_values.index <= trough_date][portfolio_values == peak_value_at_trough_peak]
                if not relevant_peaks.empty:
                    drawdown_start_date = relevant_peaks.index[0]
                else: # Should not happen if logic is correct, but as a fallback
                    drawdown_start_date = pd.NaT 

                print(f"\nWorst Drawdown for {portfolio_name}: {worst_drawdown_pct*100:.2f}% from {drawdown_start_date.strftime('%Y-%m-%d') if pd.notna(drawdown_start_date) else 'N/A'} to {trough_date.strftime('%Y-%m-%d')}")

                # Calculate performance of other portfolios during this drawdown period
                other_portfolio_performance = {}
                if pd.notna(drawdown_start_date) and pd.notna(trough_date) and drawdown_start_date <= trough_date:
                    for other_portfolio in daily_cumulative_growth.columns:
                        if other_portfolio != portfolio_name:
                            start_val = daily_cumulative_growth.loc[drawdown_start_date, other_portfolio]
                            end_val = daily_cumulative_growth.loc[trough_date, other_portfolio]
                            if pd.notna(start_val) and pd.notna(end_val) and start_val != 0:
                                performance = ((end_val / start_val) - 1) * 100
                                other_portfolio_performance[f'{other_portfolio}_Perf (%)'] = round(performance, 2)
                            else:
                                other_portfolio_performance[f'{other_portfolio}_Perf (%)'] = np.nan
                
                drawdown_info = {
                    'Portfolio with Max Drawdown': portfolio_name,
                    'Max Drawdown (%)': round(worst_drawdown_pct * 100, 2),
                    'Drawdown Start Date (Peak)': drawdown_start_date.strftime('%Y-%m-%d') if pd.notna(drawdown_start_date) else None,
                    'Drawdown End Date (Trough)': trough_date.strftime('%Y-%m-%d') if pd.notna(trough_date) else None,
                }
                drawdown_info.update(other_portfolio_performance)
                all_max_drawdown_data.append(drawdown_info)

            else:
                print(f"\nWorst Drawdown for {portfolio_name}: Not available (drawdown series is all NaN).")
        else:
            print(f"\nWorst Drawdown for {portfolio_name}: Not available (no portfolio value data).")

    # Save extreme days to CSV
    if all_extreme_days_data:
        extreme_days_df = pd.DataFrame(all_extreme_days_data)
        csv_filename_extreme = 'extreme_performing_days.csv'
        try:
            extreme_days_df.to_csv(csv_filename_extreme, index=False)
            print(f"\nExtreme performing days saved to {csv_filename_extreme}")
        except Exception as e:
            print(f"Error saving CSV file {csv_filename_extreme}: {e}")

    # Save max drawdown analysis to CSV
    if all_max_drawdown_data:
        max_drawdown_df = pd.DataFrame(all_max_drawdown_data)
        csv_filename_drawdown = 'max_drawdown_analysis.csv'
        try:
            # Reorder columns for clarity
            cols_order = ['Portfolio with Max Drawdown', 'Max Drawdown (%)', 'Drawdown Start Date (Peak)', 'Drawdown End Date (Trough)']
            other_perf_cols = [col for col in max_drawdown_df.columns if col not in cols_order]
            max_drawdown_df = max_drawdown_df[cols_order + sorted(other_perf_cols)]
            max_drawdown_df.to_csv(csv_filename_drawdown, index=False)
            print(f"Max drawdown analysis saved to {csv_filename_drawdown}")
        except Exception as e:
            print(f"Error saving CSV file {csv_filename_drawdown}: {e}")
            
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
    qqq_file = 'QQQ ETF Stock Price History.csv' # Replace with your actual file name/path
    spy_file = 'SPY ETF Stock Price History.csv' # Replace with your actual file name/path
    iwm_file = 'IWM ETF Stock Price History.csv' # Replace with your actual file name/path

    # Load data
    qqq_data = load_and_preprocess_data(qqq_file, 'QQQ')
    spy_data = load_and_preprocess_data(spy_file, 'SPY')
    iwm_data = load_and_preprocess_data(iwm_file, 'IWM')

    if qqq_data is None or spy_data is None or iwm_data is None:
        print("Exiting due to data loading errors. Please ensure CSV files are present and correctly named/located.")
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