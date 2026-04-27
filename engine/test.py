from data import fetch
from strategy import EMACrossover, RSIMeanReversion, BollingerBreakout
from backtest import Backtest

df = fetch('AAPL', '2020-01-01', '2021-01-01')

strategies = [
    EMACrossover(12, 26),
    RSIMeanReversion(30, 70),
    BollingerBreakout(20, 2)
]

for strat in strategies:
    entries, exits = strat.generate_signals(df)
    print(f"{strat.__class__.__name__}")
    print(f"Entries: {entries.sum()}")
    print(f"Exits: {exits.sum()}")
