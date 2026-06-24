import yfinance as yf
import pandas as pd


def fetch_real_market_data(tickers, lookback_period):
    """
    Fetch historical market data for given tickers using yfinance.
    
    Args:
        tickers (list): List of ticker symbols to fetch data for
        lookback_period (int): Number of days of historical data to fetch
    
    Returns:
        pd.DataFrame: DataFrame containing daily returns for each ticker
    """
    # Download historical data
    data = yf.download(tickers, period=f"{lookback_period}d", progress=False)
    
    # Handle different data structures from yfinance
    if len(tickers) > 1:
        # Multi-ticker case: yfinance returns MultiIndex columns with structure (Price, Ticker)
        if isinstance(data.columns, pd.MultiIndex):
            # Check column level names to determine structure
            # New yfinance versions use ('Price', 'Ticker') structure
            if data.columns.names == ['Price', 'Ticker']:
                # Extract Close prices for all tickers
                prices = data.xs('Close', level='Price', axis=1)
            elif 'Adj Close' in data.columns.get_level_values(0):
                # Older structure with 'Adj Close' as first level
                prices = data['Adj Close']
            elif 'Close' in data.columns.get_level_values(0):
                # Older structure with 'Close' as first level
                prices = data['Close']
            else:
                # Fallback: use the first numeric column group
                prices = data.iloc[:, :len(tickers)]
        else:
            # Single-level columns with multi-ticker data
            prices = data
    else:
        # Single ticker case
        if isinstance(data, pd.DataFrame):
            if 'Adj Close' in data.columns:
                prices = data[['Adj Close']]
                prices.columns = tickers
            elif 'Close' in data.columns:
                prices = data[['Close']]
                prices.columns = tickers
            else:
                prices = data
        else:
            # If data is a Series, convert to DataFrame
            prices = pd.DataFrame(data)
            prices.columns = tickers
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    return returns
