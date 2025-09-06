import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta 

print("=== Portfolio Construction and Rebalancing Project ===\n")
np.random.seed(42)  

stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA'] 
print(f"Selected stocks for our portfolio: {stocks}")
start_date = '2022-01-01' 
end_date = '2023-12-31'   
dates = pd.date_range(start=start_date, end=end_date, freq='D') 

print(f"Analysis period: {start_date} to {end_date}")
print(f"Total days in analysis: {len(dates)} days\n")

initial_prices = {
    'AAPL': 150.0,  
    'MSFT': 300.0,   
    'GOOGL': 2800.0, 
    'AMZN': 3200.0,  
    'TSLA': 1000.0, 
    'NVDA': 250.0   
}
stock_data = pd.DataFrame(index=dates)  
for stock in stocks:
    prices = [initial_prices[stock]] 
    daily_returns = np.random.normal(0.0008, 0.02, len(dates)-1) 
    for return_rate in daily_returns:
        new_price = prices[-1] * (1 + return_rate) 
        prices.append(new_price) 
    
    stock_data[stock] = prices  

print("Sample of generated stock prices:")
print(stock_data.head())  
print(f"\nStock data shape: {stock_data.shape}")  

print("\n=== Calculating Daily Returns ===")

returns_data = stock_data.pct_change().dropna()

print("Sample of daily returns (first 5 days):")
print(returns_data.head()) 
print(f"Returns data shape: {returns_data.shape}")

print("\nDaily Returns Statistics:")
print(returns_data.describe())  

print("\n=== Portfolio Construction ===")

num_stocks = len(stocks)  
equal_weight = 1.0 / num_stocks 

print(f"Number of stocks in portfolio: {num_stocks}")
print(f"Equal weight per stock: {equal_weight:.4f} ({equal_weight*100:.2f}%)")

initial_weights = np.array([equal_weight] * num_stocks) 
weights_df = pd.DataFrame(index=stocks, columns=['Weight']) 
weights_df['Weight'] = initial_weights  

print("\nInitial Portfolio Weights:")
print(weights_df)
print(f"Total weight (should be 1.0): {weights_df['Weight'].sum():.4f}")

print("\n=== Calculating Portfolio Performance ===")

portfolio_returns = (returns_data * initial_weights).sum(axis=1) 

print("Sample portfolio returns (first 10 days):")
print(portfolio_returns.head(10))

portfolio_cumulative = (1 + portfolio_returns).cumprod() 
individual_cumulative = (1 + returns_data).cumprod()     

print(f"\nPortfolio total return: {(portfolio_cumulative.iloc[-1] - 1)*100:.2f}%")

print("\n=== Implementing Quarterly Rebalancing ===")

quarterly_dates = [] 
current_date = pd.Timestamp(start_date) 
end_ts = pd.Timestamp(end_date) 
while current_date <= end_ts:    
    quarter_end = current_date + pd.offsets.QuarterEnd(0) 
    if quarter_end <= end_ts and quarter_end in stock_data.index:
        quarterly_dates.append(quarter_end) 
    current_date = quarter_end + pd.DateOffset(days=1)

print(f"Rebalancing dates ({len(quarterly_dates)} times):")
for date in quarterly_dates:
    print(f"  {date.date()}")

rebalanced_portfolio_value = []  
rebalanced_dates = [] 

initial_investment = 100000 
current_value = initial_investment 

print(f"\nStarting portfolio value: ${initial_investment:,.2f}")
for i, rebal_date in enumerate(quarterly_dates):
    if i == 0:        
        period_start = returns_data.index[0] 
    else:
        
        period_start = quarterly_dates[i-1] 
    
    period_mask = (returns_data.index > period_start) & (returns_data.index <= rebal_date)
    period_returns = returns_data.loc[period_mask] 
    
    if len(period_returns) > 0:    
        period_portfolio_returns = (period_returns * initial_weights).sum(axis=1)
        period_cumulative = (1 + period_portfolio_returns).cumprod()
        
        if len(period_cumulative) > 0:
            current_value *= period_cumulative.iloc[-1] 
        rebalanced_portfolio_value.append(current_value) 
        rebalanced_dates.append(rebal_date) 
        
        print(f"Portfolio value on {rebal_date.date()}: ${current_value:,.2f}")

print(f"\nFinal portfolio value: ${current_value:,.2f}")
print(f"Total return with rebalancing: {((current_value/initial_investment)-1)*100:.2f}%")

print("\n=== Performance Analysis ===")
trading_days = len(returns_data) 
years = trading_days / 252 

portfolio_total_return = portfolio_cumulative.iloc[-1] - 1 
portfolio_annualized = (1 + portfolio_total_return) ** (1/years) - 1  

print(f"Analysis period: {years:.2f} years")
print(f"Portfolio annualized return: {portfolio_annualized*100:.2f}%")

print("\nIndividual Stock Performance:")
for stock in stocks:
    stock_total_return = individual_cumulative[stock].iloc[-1] - 1 
    stock_annualized = (1 + stock_total_return) ** (1/years) - 1   
    print(f"{stock}: {stock_annualized*100:.2f}% annualized return")

portfolio_volatility = portfolio_returns.std() * np.sqrt(252) 
print(f"\nPortfolio annualized volatility: {portfolio_volatility*100:.2f}%")

risk_free_rate = 0.02 
sharpe_ratio = (portfolio_annualized - risk_free_rate) / portfolio_volatility
print(f"Portfolio Sharpe ratio: {sharpe_ratio:.3f}")

print("\n=== Creating Visualizations ===")

plt.figure(figsize=(10, 6))
plt.plot(stock_data.index, stock_data)
plt.title('Stock Prices Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend(stocks, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(individual_cumulative.index, individual_cumulative)
plt.plot(portfolio_cumulative.index, portfolio_cumulative, 'black', linewidth=3, label='Portfolio')
plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (Starting at $1)')
plt.legend(stocks + ['Portfolio'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()

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

plt.figure(figsize=(8, 8))
plt.pie(initial_weights, labels=stocks, autopct='%1.1f%%', startangle=90)
plt.title('Portfolio Allocation (Equal Weighted)', fontsize=14, fontweight='bold')
plt.show()

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
best_stock = individual_cumulative.iloc[-1].idxmax() 
best_return = individual_cumulative[best_stock].iloc[-1] - 1  
print(f"{best_stock}: {best_return*100:.2f}% total return")

print(f"\nWORST PERFORMING STOCK:")
worst_stock = individual_cumulative.iloc[-1].idxmin() 
worst_return = individual_cumulative[worst_stock].iloc[-1] - 1 
print(f"{worst_stock}: {worst_return*100:.2f}% total return")

print(f"\nPORTFOLIO vs INDIVIDUAL STOCKS:")
avg_individual_return = individual_cumulative.iloc[-1].mean() - 1  
print(f"Average individual stock return: {avg_individual_return*100:.2f}%")
print(f"Portfolio return: {portfolio_total_return*100:.2f}%")

if portfolio_total_return > avg_individual_return:
    print(" Portfolio OUTPERFORMED average individual stock return")
else:
    print(" Portfolio UNDERPERFORMED average individual stock return")

print("\n" + "="*60)
print("Analysis Complete! You've successfully built and analyzed a quantitative portfolio.")
print("="*60)
