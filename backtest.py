import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotStrat(strat):
    plt.plot(np.exp(strat.cumsum())-1)
    plt.grid(True)
    plt.show(block=False)

def calc_sharpe(strat: pd.DataFrame, n: int = 252):
    temp = np.exp(strat) - 1
    return (temp.mean())/temp.std() * np.sqrt(n)

def calc_max_drawdown(strat: pd.DataFrame):
    cumulative_returns = np.exp(strat.cumsum())
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    return max_drawdown

def minimum_value(strat: pd.DataFrame):
    return np.exp(strat.cumsum().min()) - 1

def maximum_value(strat: pd.DataFrame):
    return np.exp(strat.cumsum().max()) - 1

def win_rate(strat: pd.DataFrame):
    return (strat > 0).mean()/((strat > 0).mean() + (strat < 0).mean())

def average_win(strat: pd.DataFrame):
    return strat[strat > 0].mean()

def average_loss(strat: pd.DataFrame):
    return strat[strat < 0].mean()

def expected_value(strat: pd.DataFrame):
    win = win_rate(strat)
    return win * average_win(strat) + (1-win) * average_loss(strat)

def calc_stats(strat: pd.DataFrame, n=252):
    stats = {}
    stats["SR"] = calc_sharpe(strat, n)
    stats["Max Drawdown"] = calc_max_drawdown(strat)
    stats["Minimum"] = minimum_value(strat)
    stats["Maximum"] = maximum_value(strat)
    stats["Win Rate"] = win_rate(strat)
    stats["Average Win"] = average_win(strat)
    stats["Average Loss"] = average_loss(strat)
    stats["Expected Value"] = expected_value(strat)
    return pd.DataFrame({"Stats": stats})