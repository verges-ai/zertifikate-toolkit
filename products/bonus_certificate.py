import numpy as np
import pandas as pd

def simulate_price_series(n_days=100, start_price=100, volatility=0.02, barrier=90):
    prices = [start_price]
    for _ in range(n_days - 1):
        daily_return = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    prices = np.array(prices)
    barrier_violated = int(np.any(prices <= barrier))
    return prices, barrier_violated

def extract_features(prices):
    return {
        'last_price': prices[-1],
        'min_price': np.min(prices),
        'max_price': np.max(prices),
        'mean_price': np.mean(prices),
        'std_price': np.std(prices),
        'volatility': np.std(np.diff(np.log(prices + 1e-9)))  # für log-returns Volatilität
    }
