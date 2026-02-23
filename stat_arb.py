import pandas as pd
import numpy as np
from itertools import combinations
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class Stock:
    """
    This class will store the current position, value, and price of any one stock in particular
    """
    def __init__(self, ticker: str):
        self.name = ticker
        self.position = 0
        self.value = 0

    # this function represents going long or buying this stock
    def buy(self, entry_amount: float):
        self.position = entry_amount/self.price
        self.value = entry_amount
    
    # this function represents going short or selling this stock
    def sell(self, entry_amount: float):
        self.position = -entry_amount/self.price
        self.value = -entry_amount

    # this will exit all positions on this stock
    def exit(self):
        final_value = self.value
        self.position = 0
        self.value = 0
        return final_value

    # this will update the price and value of this stock
    def update_value(self, new_price):
        self.price = new_price
        self.value = self.position * self.price

    def __eq__(self, other: Stock):
        return other.name == self.name

    # this allows us to use the + operator to add two stocks together
    def __add__(self, other: Stock):
        return self.value + other.value
    
    # this allows us to use the + operator to add an int and a stock together
    def __radd__(self, other: int):
        return self.value + other

    # this allows us to use the - operator to add two stocks together
    def __sub__(self, other: Stock):
        return self.value - other.value

class Pair:
    """
    This class will store the alpha and beta from the linear regression of the two stocks, and a class for both stocks
    """
    def __init__(self, stock1: Stock, stock2: Stock, beta: float, alpha: float):
        self.stock1 = stock1
        self.stock2 = stock2
        self.z_score = 0
        self.active = False
        self.BUY = False
        self.SELL = False
        self.history = []
        self.beta = beta
        self.alpha = alpha

    # this allows us to use the + operator to add two pairs together, this will help when we calculate our portfolio value
    def __add__(self, other: Pair):
        return (self.stock1 + self.stock2) + (other.stock1 + other.stock2)
    
    # this allows us to use the + operator to add an int and a pair together, this is useful when using the 
    # sum() function to add together all pairs value.
    def __radd__(self, other: int):
        return (self.stock1 + self.stock2) + other
    
    # this allows us to use the - operator to add two pairs together, not used at the moment, but could be useful
    def __sub__(self, other: Pair):
        return (self.stock1 + self.stock2) - (other.stock1 + other.stock2)

    # this will update the stock values, and the pair values. This will also keep a history of the PnL.
    def update(self, z_score: float, prices: dict[str: float]):
        prev_value = self.stock1 + self.stock2
        self.stock1.update_value(prices[self.stock1.name])
        self.stock2.update_value(prices[self.stock2.name])
        self.z_score = z_score
        
        # if the trade is active, then calcuate the period return or PnL for this day.
        if self.active:
            period_return = (self.stock1 + self.stock2) - prev_value - (2 * self.t_costs)
            # after calculating the trasaction costs for this trade, set it to 0 so it is not counted on every day.
            self.t_costs = 0
        else:
            period_return = 0
        
        self.history.append(period_return)

    # This represents going long on stock1 and short on stock2
    def buy(self, entry_amount: float, t_costs):
        self.stock1.buy(entry_amount)
        self.stock2.sell(entry_amount)
        self.active = True
        self.BUY = True
        self.t_costs = t_costs * entry_amount
    
    # This represents going short on stock1 and long on stock2
    def sell(self, entry_amount: float, t_costs):
        self.stock1.sell(entry_amount)
        self.stock2.buy(entry_amount)
        self.active = True
        self.SELL = True
        self.t_costs = t_costs * entry_amount

    # this represents exiting all positions
    def exit(self, t_costs=0) -> float:
        value = self.stock1.exit()
        value += self.stock2.exit()
        self.active = False
        self.BUY = False
        self.SELL = False
        return value

    # this is the string magic method, this will be used to get this pairs update information 
    # including the new stock prices and z_scores.
    def __str__(self):
        return f"{self.stock1.name}-{self.stock2.name}"


class Strategy:
    def __init__(self, balance = 10000, usage_per_pair = .05, t_costs = 0.001):
        self.upp = usage_per_pair
        self.pairs: list[Pair] = []
        self.balance = balance
        self.stock_value = 0
        self.value = self.stock_value + self.balance
        self.value_history = []
        self.t_costs = t_costs

    def addPair(self, pair: Pair):
        self.pairs.append(pair)
    
    def update(self, z_scores: dict[str: float], prices: pd.DataFrame):
        for pair in self.pairs:
            name = f"{pair}"
            pair.update(z_scores[name], prices[name].to_dict())
        self.stock_value = sum(self.pairs)
        self.value = self.balance + self.stock_value
        self.value_history.append(self.value)
    
    def execute(self, threshold: float = 2.0):
        for pair in self.pairs:
            if pair.active:
                if (pair.SELL and pair.z_score < 0) or (pair.BUY and pair.z_score > 0):
                    value = pair.exit()
                    self.balance += value
                    # self.stock_value -= value
            else:
                if pair.z_score > threshold:
                    pair.sell(self.balance * self.upp, self.t_costs)
                elif pair.z_score < -threshold:
                    pair.buy(self.balance * self.upp, self.t_costs)

# this function will run the extensive backtest calculating PnL and return the log returns of the equity curve
def stat_arb_backtest(data, stock_pairs, n = 50, initial_balance=10000, t_costs=0.001, leverage=1, z_entry: float = 2.0) -> pd.DataFrame:
    # this will get all of the stock pairs for the backtest
    pairs = stock_pairs[["Stock1", "Stock2"]].to_numpy().tolist()

    # this will get all the stock pairs and their data for the backtest
    pairs2 = stock_pairs.to_numpy().tolist()

    # make a straegy class and add all stock pairs to it, first you will need to make stock objects, 
    # then pair objects, then add the pairs to the class.
    class_strat = Strategy(initial_balance, 1.0/len(pairs) * leverage, t_costs)
    stocks = np.array(pairs).flatten().astype(str).tolist()
    stock_dict = {}
    for stock in stocks:
        stock_dict[stock] = Stock(stock)

    for stock1, stock2, corr, adf, p, beta, alpha in pairs2:
        class_strat.addPair(Pair(stock_dict[stock1], stock_dict[stock2], beta, alpha))

    # make a dictionary of DataFrames that maps the name of pairs to time_series data of z_scores, and prices
    zscores = pd.DataFrame(index = data.index)
    stocks_dict = {}
    for stock1, stock2 in pairs:
        spread = data[stock1] - data[stock2]
        zscores[f"{stock1}-{stock2}"] = (spread - spread.rolling(n).mean()) / spread.rolling(n).std()
        stocks_dict[f"{stock1}-{stock2}"] = data[[stock1, stock2]]

    # Loop over all timsteps in the backtest and update the prices and z_scores, then execute the strategy.
    zscores.fillna(0, inplace=True)
    stocks_df: pd.DataFrame = pd.concat(stocks_dict, axis=1).fillna(0)
    for i in range(len(stocks_df)):
        class_strat.update(zscores.iloc[i].to_dict(), stocks_df.iloc[i])
        class_strat.execute(z_entry)

    # get the PnL of each stock pair.
    temp_df = pd.DataFrame(index=data.index)
    for pair in class_strat.pairs:
        temp_df[f"{pair}"] = np.array(pair.history)

    # calculate the equity curve, and return the log returns of that.
    equity_curve = 10000 + temp_df.sum(axis=1).cumsum()
    return np.log(equity_curve / equity_curve.shift()).fillna(0)

# this function will do linear regression on two stocks, this is not being used as timeseries data anymore, 
# rather as one stock price is a function of the other.
def do_reg(data, stock1, stock2):
    # linear regression
    X = data[stock1].values
    y = data[stock2].values

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    alpha, beta = model.params
    res = y - (alpha + beta * X[:,1])

    # adfuller test will fail in the result of nan values
    if np.isnan(res).any():
        return np.nan
    
    # run an adfuller cointegration test on the residual of the linear regression. The more negitive a score is, 
    # the more mean reverting a pair should be.
    adf = adfuller(res)

    # return the important values from the regression and adfuller test.
    return adf[0], adf[1], beta, alpha
    
# this function will take a DataFrame of time series stock prices and return a DataFrame of all stock pairs 
# that are coorilated along with their cointegration test results.
def get_pairs(data: pd.DataFrame):
    df_comb = pd.DataFrame(combinations(data.columns, 2), columns=["Stock1", "Stock2"])
    df_comb["corr"] = df_comb.apply(lambda row: np.corrcoef(data[row["Stock1"]], data[row["Stock2"]])[0,1], axis=1)
    df_comb = df_comb[df_comb["corr"] > 0.95]
    df_comb[["adf", "p_value", "beta", "alpha"]] = df_comb.apply(lambda row: pd.Series(do_reg(data, row["Stock1"], row["Stock2"])), axis=1)
    return df_comb

def main():
    pass

if __name__ == "__main__":
    main()