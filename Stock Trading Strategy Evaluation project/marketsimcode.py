
import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
#from TheoreticallyOptimalStrategy import testPolicy

def author():
    return "zelahmadi3"

def compute_portvals(
        df,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    orders_filtered = df[df!=0].dropna()
    orders= orders_filtered.copy()
    orders[orders_filtered > 0] = "BUY"
    orders[orders_filtered < 0] = "SELL"
    orders.rename(columns={orders.columns[0]: "Order"}, inplace=True)
    orders["Symbol"] = orders_filtered.columns[0]
    orders["Shares"] = abs(orders_filtered.iloc[:, 0])

    sd = orders.index.min()
    ed = orders.index.max()
    symbols = list(set(orders['Symbol']))
    dates = pd.date_range(sd, ed)

    S_prices = get_data(symbols, dates, addSPY=True, colname="Adj Close")
    S_prices= S_prices[symbols]
    Prices = S_prices.copy()

    Prices.sort_index(inplace=True)  # we sort the dates prices for our stocks sorted

    Prices["Cash"] = 1  # adding cash column and initialize it with 1

    Trades = Prices.copy()  # make a copy of Prices data frame to hold trades on each trading day
    Trades[:] = 0.0  # initialize all trades with zero
    T = Prices.copy()  # creating T data frame to hold transaction cost of each day including impact and commission
    T[:] = 0.0  # initialize it with zero
    for i, j in enumerate(orders.index.strftime("%Y-%m-%d").tolist()):  # looping through indexes of orders
        stock = orders['Symbol'][i]
        if orders["Order"][i] == "SELL":
            Trades.loc[j, stock] += -orders['Shares'][i]  # track number of shares sold
            T.loc[j, stock] += orders['Shares'][i] * Prices.loc[j, stock] * (1 - impact) - commission
        else:

            Trades.loc[j, stock] += orders['Shares'][i]  # track number of shares bought
            T.loc[j, stock] += -orders['Shares'][i] * Prices.loc[j, stock] * (1 + impact) - commission

    Trades["Cash"] = T.sum(axis=1)  # the total transaction cost for all trades on each day
    holding = Trades.copy()
    holding[:] = 0.0
    holding["Cash"][0] = start_val  # initial cash value
    holding.iloc[0] = Trades.iloc[0] + holding.iloc[0]

    for i in range(1, holding.shape[0]):
        holding.iloc[i] = Trades.iloc[i] + holding.iloc[i - 1]  # summing the current amount with the previous one

    values = Prices * holding  # calculating the value of stock held
    portfolio_val = values.sum(axis=1)  # the total value of the portfolio stock and cash

    return portfolio_val

