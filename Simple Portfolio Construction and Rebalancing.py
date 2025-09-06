# Simple Portfolio Construction and Rebalancing Project
# A beginner-friendly quantitative finance project

import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical calculations
import matplotlib.pyplot as plt  # For creating visualizations
from datetime import datetime, timedelta  # For date operations

# Step 1: Create Sample Stock Data (Simulating Real Market Data)
print("=== Portfolio Construction and Rebalancing Project ===\n")

# Set random seed for reproducible results
np.random.seed(42)  # Ensures same results every time we run the code

# Define our stock universe - 6 popular stocks
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']  # List of stock symbols
print(f"Selected stocks for our portfolio: {stocks}")

# Create date range for our analysis (2 years of daily data)
start_date = '2022-01-01'  # Portfolio start date
end_date = '2023-12-31'    # Portfolio end date
dates = pd.date_range(start=start_date, end=end_date, freq='D')  # Daily frequency

print(f"Analysis period: {start_date} to {end_date}")
print(f"Total days in analysis: {len(dates)} days\n")

# Step 2: Generate Realistic Stock Price Data
# We'll simulate stock prices that behave somewhat like real stocks

# Starting prices for each stock (in dollars)
initial_prices = {
    'AAPL': 150.0,   # Apple starting price
    'MSFT': 300.0,   # Microsoft starting price  
    'GOOGL': 2800.0, # Google starting price
    'AMZN': 3200.0,  # Amazon starting price
    'TSLA': 1000.0,  # Tesla starting price
    'NVDA': 250.0    # Nvidia starting price
}

# Create empty DataFrame to store all stock prices
stock_data = pd.DataFrame(index=dates)  # DataFrame with dates as index

# Generate price data for each stock using random walk simulation
for stock in stocks:
    prices = [initial_prices[stock]]  # Start with initial price
    
    # Generate daily returns (percentage changes) - normally distributed
    daily_returns = np.random.normal(0.0008, 0.02, len(dates)-1)  # Mean=0.08% daily, Std=2%
    
    # Convert daily returns to actual prices using compound growth
    for return_rate in daily_returns:
        new_price = prices[-1] * (1 + return_rate)  # Calculate next day's price
        prices.append(new_price)  # Add to price list
    
    stock_data[stock] = prices  # Add stock prices to DataFrame

print("Sample of generated stock prices:")
print(stock_data.head())  # Show first 5 rows of data
print(f"\nStock data shape: {stock_data.shape}")  # Show dimensions (rows, columns)

# Step 3: Calculate Daily Returns for Each Stock
print("\n=== Calculating Daily Returns ===")

# Calculate percentage change from previous day for each stock
returns_data = stock_data.pct_change().dropna()  # pct_change() calculates daily returns, dropna() removes NaN values

print("Sample of daily returns (first 5 days):")
print(returns_data.head())  # Show first 5 rows of returns
print(f"Returns data shape: {returns_data.shape}")

# Calculate and display basic statistics for returns
print("\nDaily Returns Statistics:")
print(returns_data.describe())  # Shows count, mean, std, min, max, quartiles

# Step 4: Portfolio Construction - Equal Weighted
print("\n=== Portfolio Construction ===")

# Equal-weighted portfolio means each stock gets same allocation
num_stocks = len(stocks)  # Number of stocks in portfolio
equal_weight = 1.0 / num_stocks  # Each stock gets 1/6 = 16.67% allocation

print(f"Number of stocks in portfolio: {num_stocks}")
print(f"Equal weight per stock: {equal_weight:.4f} ({equal_weight*100:.2f}%)")

# Create initial portfolio weights
initial_weights = np.array([equal_weight] * num_stocks)  # Array of equal weights
weights_df = pd.DataFrame(index=stocks, columns=['Weight'])  # DataFrame for weights
weights_df['Weight'] = initial_weights  # Assign equal weights

print("\nInitial Portfolio Weights:")
print(weights_df)
print(f"Total weight (should be 1.0): {weights_df['Weight'].sum():.4f}")

# Step 5: Calculate Portfolio Returns
print("\n=== Calculating Portfolio Performance ===")

# Portfolio return = sum of (individual stock return × weight)
portfolio_returns = (returns_data * initial_weights).sum(axis=1)  # Weighted average of returns

print("Sample portfolio returns (first 10 days):")
print(portfolio_returns.head(10))

# Calculate cumulative returns (portfolio value over time)
portfolio_cumulative = (1 + portfolio_returns).cumprod()  # Compound returns over time
individual_cumulative = (1 + returns_data).cumprod()      # Individual stock cumulative returns

print(f"\nPortfolio total return: {(portfolio_cumulative.iloc[-1] - 1)*100:.2f}%")

# Step 6: Implement Quarterly Rebalancing
print("\n=== Implementing Quarterly Rebalancing ===")

# Find quarter-end dates for rebalancing
quarterly_dates = []  # List to store rebalancing dates
current_date = pd.Timestamp(start_date)  # Start from beginning
end_ts = pd.Timestamp(end_date)  # End timestamp

# Generate quarterly rebalancing dates
while current_date <= end_ts:
    # Find last business day of each quarter
    quarter_end = current_date + pd.offsets.QuarterEnd(0)  # End of current quarter
    if quarter_end <= end_ts and quarter_end in stock_data.index:
        quarterly_dates.append(quarter_end)  # Add to rebalancing dates
    current_date = quarter_end + pd.DateOffset(days=1)  # Move to next quarter

print(f"Rebalancing dates ({len(quarterly_dates)} times):")
for date in quarterly_dates:
    print(f"  {date.date()}")

# Simulate portfolio with quarterly rebalancing
rebalanced_portfolio_value = []  # Track portfolio value with rebalancing
rebalanced_dates = []  # Track corresponding dates

# Start with $100,000 initial investment
initial_investment = 100000  # Starting portfolio value
current_value = initial_investment  # Current portfolio value

print(f"\nStarting portfolio value: ${initial_investment:,.2f}")

# Calculate returns between rebalancing dates
for i, rebal_date in enumerate(quarterly_dates):
    if i == 0:
        # First rebalancing - start from beginning of data
        period_start = returns_data.index[0]  # First date in returns data
    else:
        # Subsequent rebalancing - start from previous rebalancing date
        period_start = quarterly_dates[i-1]  # Previous rebalancing date
    
    # Get returns for this period
    period_mask = (returns_data.index > period_start) & (returns_data.index <= rebal_date)
    period_returns = returns_data.loc[period_mask]  # Returns for this period
    
    if len(period_returns) > 0:
        # Calculate portfolio performance for this period
        period_portfolio_returns = (period_returns * initial_weights).sum(axis=1)
        period_cumulative = (1 + period_portfolio_returns).cumprod()
        
        # Update portfolio value
        if len(period_cumulative) > 0:
            current_value *= period_cumulative.iloc[-1]  # Compound the returns
        
        rebalanced_portfolio_value.append(current_value)  # Record portfolio value
        rebalanced_dates.append(rebal_date)  # Record date
        
        print(f"Portfolio value on {rebal_date.date()}: ${current_value:,.2f}")

print(f"\nFinal portfolio value: ${current_value:,.2f}")
print(f"Total return with rebalancing: {((current_value/initial_investment)-1)*100:.2f}%")

# Step 7: Performance Comparison and Analysis
print("\n=== Performance Analysis ===")

# Calculate annualized returns
trading_days = len(returns_data)  # Number of trading days
years = trading_days / 252  # Convert to years (252 trading days per year)

# Portfolio annualized return
portfolio_total_return = portfolio_cumulative.iloc[-1] - 1  # Total return
portfolio_annualized = (1 + portfolio_total_return) ** (1/years) - 1  # Annualized return

print(f"Analysis period: {years:.2f} years")
print(f"Portfolio annualized return: {portfolio_annualized*100:.2f}%")

# Calculate individual stock annualized returns
print("\nIndividual Stock Performance:")
for stock in stocks:
    stock_total_return = individual_cumulative[stock].iloc[-1] - 1  # Stock total return
    stock_annualized = (1 + stock_total_return) ** (1/years) - 1    # Stock annualized return
    print(f"{stock}: {stock_annualized*100:.2f}% annualized return")

# Calculate portfolio volatility (risk measure)
portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
print(f"\nPortfolio annualized volatility: {portfolio_volatility*100:.2f}%")

# Calculate Sharpe ratio (return per unit of risk)
risk_free_rate = 0.02  # Assume 2% risk-free rate
sharpe_ratio = (portfolio_annualized - risk_free_rate) / portfolio_volatility
print(f"Portfolio Sharpe ratio: {sharpe_ratio:.3f}")

# Step 8: Visualization
print("\n=== Creating Visualizations ===")

# --- STEP 8: VISUALIZATION (4 Separate Figures) ---

# Plot 1: Stock Prices Over Time
plt.figure(figsize=(10, 6))
plt.plot(stock_data.index, stock_data)
plt.title('Stock Prices Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend(stocks, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()

# Plot 2: Cumulative Returns Comparison
plt.figure(figsize=(10, 6))
plt.plot(individual_cumulative.index, individual_cumulative)
plt.plot(portfolio_cumulative.index, portfolio_cumulative, 'black', linewidth=3, label='Portfolio')
plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (Starting at $1)')
plt.legend(stocks + ['Portfolio'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()

# Plot 3: Daily Returns Distribution
plt.figure(figsize=(10, 6))
plt.hist(portfolio_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(portfolio_returns.mean(), color='red', linestyle='--',
            label=f'Mean: {portfolio_returns.mean():.4f}')
plt.title('Portfolio Daily Returns Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot 4: Portfolio Allocation (Pie Chart)
plt.figure(figsize=(8, 8))
plt.pie(initial_weights, labels=stocks, autopct='%1.1f%%', startangle=90)
plt.title('Portfolio Allocation (Equal Weighted)', fontsize=14, fontweight='bold')
plt.show()

# Step 9: Summary Report
print("\n" + "="*60)
print("PORTFOLIO ANALYSIS SUMMARY REPORT")
print("="*60)

print(f"Portfolio Period: {start_date} to {end_date}")
print(f"Number of Stocks: {num_stocks}")
print(f"Portfolio Strategy: Equal Weighted")
print(f"Rebalancing Frequency: Quarterly ({len(quarterly_dates)} times)")

print(f"\nPERFORMANCE METRICS:")
print(f"Total Return: {portfolio_total_return*100:.2f}%")
print(f"Annualized Return: {portfolio_annualized*100:.2f}%")
print(f"Annualized Volatility: {portfolio_volatility*100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")

print(f"\nBEST PERFORMING STOCK:")
best_stock = individual_cumulative.iloc[-1].idxmax()  # Stock with highest return
best_return = individual_cumulative[best_stock].iloc[-1] - 1  # Best stock return
print(f"{best_stock}: {best_return*100:.2f}% total return")

print(f"\nWORST PERFORMING STOCK:")
worst_stock = individual_cumulative.iloc[-1].idxmin()  # Stock with lowest return
worst_return = individual_cumulative[worst_stock].iloc[-1] - 1  # Worst stock return
print(f"{worst_stock}: {worst_return*100:.2f}% total return")

print(f"\nPORTFOLIO vs INDIVIDUAL STOCKS:")
avg_individual_return = individual_cumulative.iloc[-1].mean() - 1  # Average individual return
print(f"Average individual stock return: {avg_individual_return*100:.2f}%")
print(f"Portfolio return: {portfolio_total_return*100:.2f}%")

if portfolio_total_return > avg_individual_return:
    print(" Portfolio OUTPERFORMED average individual stock return")
else:
    print(" Portfolio UNDERPERFORMED average individual stock return")

print("\n" + "="*60)
print("Analysis Complete! You've successfully built and analyzed a quantitative portfolio.")
print("="*60)