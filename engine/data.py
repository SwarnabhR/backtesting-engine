import pandas as pd
import yfinance as yf
from typing import Optional

def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {e}")
    if df.empty:
        raise ValueError(f"No data found for {ticker} between {start} and {end}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    df.columns = [col.lower() for col in df.columns]
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[required_cols]
    df = df.ffill().bfill()
    df = df.dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df