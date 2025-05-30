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
    analyzes max drawdown periods, saves results to a Markdown report file,
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

    report_sections_md = ["# Portfolio Performance Report\n\n"]
    print("\n\n--- Generating Portfolio Performance Report ---")

    for portfolio_name in daily_cumulative_growth.columns:
        current_portfolio_md_parts = [f"## Portfolio: {portfolio_name}\n\n"]
        print(f"\n--- Analyzing Metrics for {portfolio_name} ---")

        # Best/Worst Days
        current_portfolio_md_parts.append("### Extreme Performing Days\n\n")
        if portfolio_name in actual_daily_portfolio_returns.columns and actual_daily_portfolio_returns[portfolio_name].notna().any():
            daily_returns_pct = actual_daily_portfolio_returns[portfolio_name] * 100
            daily_returns_pct_cleaned = daily_returns_pct.replace([np.inf, -np.inf], np.nan).dropna()

            if not daily_returns_pct_cleaned.empty:
                best_days_series = daily_returns_pct_cleaned.nlargest(7)
                worst_days_series = daily_returns_pct_cleaned.nsmallest(7)

                best_days_df = pd.DataFrame({
                    'Date': best_days_series.index.strftime('%Y-%m-%d'),
                    'Performance (%)': best_days_series.values
                }).round({'Performance (%)': 2})
                worst_days_df = pd.DataFrame({
                    'Date': worst_days_series.index.strftime('%Y-%m-%d'),
                    'Performance (%)': worst_days_series.values
                }).round({'Performance (%)': 2})

                current_portfolio_md_parts.append("#### Top 7 Best Days\n")
                current_portfolio_md_parts.append(best_days_df.to_markdown(index=False) + "\n\n")
                current_portfolio_md_parts.append("#### Top 7 Worst Days\n")
                current_portfolio_md_parts.append(worst_days_df.to_markdown(index=False) + "\n\n")
                
                print("Top 7 Best Days (Actual Daily % Change of Portfolio Value):")
                print(best_days_series.round(2))
                print("\nTop 7 Worst Days (Actual Daily % Change of Portfolio Value):")
                print(worst_days_series.round(2))
            else:
                no_data_msg = "No valid daily returns data for best/worst days after cleaning.\n\n"
                current_portfolio_md_parts.append(no_data_msg)
                print(f"Best/Worst Days for {portfolio_name}: {no_data_msg.strip()}")
        else:
            no_data_msg = "No daily returns data available for this portfolio to calculate best/worst days.\n\n"
            current_portfolio_md_parts.append(no_data_msg)
            print(f"Best/Worst Days for {portfolio_name}: {no_data_msg.strip()}")

        # Max Drawdown Analysis
        current_portfolio_md_parts.append("### Maximum Drawdown Analysis\n\n")
        portfolio_values = daily_cumulative_growth[portfolio_name]
        if not portfolio_values.empty and portfolio_values.notna().any():
            rolling_max = portfolio_values.cummax()
            drawdown_series_val = pd.Series(index=portfolio_values.index, dtype=float)

            for date_idx in portfolio_values.index:
                peak_val_for_day = rolling_max[date_idx]
                current_val = portfolio_values[date_idx]
                if pd.isna(current_val) or pd.isna(peak_val_for_day):
                    drawdown_series_val[date_idx] = np.nan
                elif peak_val_for_day == 0:
                    drawdown_series_val[date_idx] = 0.0
                else:
                    drawdown_series_val[date_idx] = (current_val / peak_val_for_day) - 1.0
            
            if drawdown_series_val.notna().any() and not drawdown_series_val.isnull().all():
                worst_drawdown_pct = drawdown_series_val.min()
                trough_date = drawdown_series_val.idxmin()
                peak_value_at_trough_peak = rolling_max[trough_date]
                
                relevant_peaks = portfolio_values[(portfolio_values.index <= trough_date) & (portfolio_values == peak_value_at_trough_peak)]
                drawdown_start_date = relevant_peaks.index[0] if not relevant_peaks.empty else pd.NaT

                md_drawdown_start_date = drawdown_start_date.strftime('%Y-%m-%d') if pd.notna(drawdown_start_date) else 'N/A'
                md_trough_date = trough_date.strftime('%Y-%m-%d') if pd.notna(trough_date) else 'N/A'
                
                current_portfolio_md_parts.append(f"- **Max Drawdown:** {worst_drawdown_pct*100:.2f}%\n")
                current_portfolio_md_parts.append(f"- **Drawdown Start Date (Peak):** {md_drawdown_start_date}\n")
                current_portfolio_md_parts.append(f"- **Drawdown End Date (Trough):** {md_trough_date}\n\n")
                
                print(f"\nWorst Drawdown for {portfolio_name}: {worst_drawdown_pct*100:.2f}% from {md_drawdown_start_date} to {md_trough_date}")

                other_portfolio_perf_data = []
                if pd.notna(drawdown_start_date) and pd.notna(trough_date) and drawdown_start_date <= trough_date:
                    for other_portfolio in daily_cumulative_growth.columns:
                        if other_portfolio != portfolio_name:
                            start_val = daily_cumulative_growth.loc[drawdown_start_date, other_portfolio]
                            end_val = daily_cumulative_growth.loc[trough_date, other_portfolio]
                            performance = np.nan
                            if pd.notna(start_val) and pd.notna(end_val) and start_val != 0:
                                performance = ((end_val / start_val) - 1) * 100
                            other_portfolio_perf_data.append({
                                'Portfolio': other_portfolio,
                                'Performance During Drawdown (%)': round(performance, 2) if pd.notna(performance) else 'N/A'
                            })
                
                if other_portfolio_perf_data:
                    other_perf_df = pd.DataFrame(other_portfolio_perf_data)
                    current_portfolio_md_parts.append("#### Performance of Other Portfolios During This Drawdown:\n")
                    current_portfolio_md_parts.append(other_perf_df.to_markdown(index=False) + "\n\n")
                else:
                    current_portfolio_md_parts.append("No other portfolios to compare or period not valid for comparison.\n\n")
            else:
                no_data_msg = "Worst Drawdown: Not available (drawdown series is all NaN or empty).\n\n"
                current_portfolio_md_parts.append(no_data_msg)
                print(f"\nWorst Drawdown for {portfolio_name}: {no_data_msg.strip()}")
        else:
            no_data_msg = "Worst Drawdown: Not available (no portfolio value data).\n\n"
            current_portfolio_md_parts.append(no_data_msg)
            print(f"\nWorst Drawdown for {portfolio_name}: {no_data_msg.strip()}")
        
        current_portfolio_md_parts.append("---\n\n") # Separator
        report_sections_md.append("".join(current_portfolio_md_parts))

    # Write the combined markdown report to a file
    report_filename_md = 'portfolio_performance_report.md'
    try:
        with open(report_filename_md, 'w', encoding='utf-8') as f: # Added encoding
            f.write("".join(report_sections_md))
        print(f"\nPortfolio performance report saved to {report_filename_md}")
    except Exception as e:
        print(f"Error saving Markdown report file {report_filename_md}: {e}")
            
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
    USER_CONFIGURABLE_START_DATE = "2022-01-01"
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

    # Analyze portfolios (this will save Markdown report, print to console, and return daily growth)
    daily_cumulative_growth_df = analyze_portfolios(portfolio_returns)

    # Plot results if data is available
    if daily_cumulative_growth_df is not None and not daily_cumulative_growth_df.empty:
        plot_daily_portfolio_growth(daily_cumulative_growth_df)
    else:
        print("No daily cumulative growth data to plot.")

    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()