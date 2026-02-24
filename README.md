# Statistical Arbitrage - Pairs Trading Strategy
A pairs trading strategy using cointegration analysis across 600+ stocks, achieving a 1.55 Sharpe ratio and 4.76% max drawdown on backtests from 2018–2022.

## Overview
This project implements a statistical arbitrage strategy by identifying cointegrated stock pairs and trading the spread when it deviates from its historical mean. Transaction costs are modeled to ensure realistic performance estimates.

## Methodology
- Universe: 600+ stocks screened from S&P 500 and Russell 1000
- Pair Selection: Augmented Dickey-Fuller (ADF) cointegration tests to identify statistically valid pairs
- Signal Generation: Z-score of the spread; enter when |z| > 1.0, exit when |z| < 0.0
- Risk Management: Position sizing based on percent of portfolio, stop-loss at |z| > 3.5
- Transaction Costs: Modeled at 25 bps per trade to reflect realistic execution, including spread, slippage, and fees.

## Results
| Metric | Value |
|---|---|
| Sharpe Ratio | 1.55 |
| Max Drawdown | 4.76% | 
| Win Rate | 52.95% |
| Backtest Period | 2018–2022 |

## Tech Stack
Python, Pandas, NumPy, statsmodels, matplotlib

## Usage
Use `pip install -r requirements.txt` in the terminal to install all required dependencies, then follow the instructions in the integrated Python notebook.
