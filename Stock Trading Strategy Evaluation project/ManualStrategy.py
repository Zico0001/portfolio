import indicators as id
import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data
#from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals
class ManualStrategy:

    def testPolicy(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates, addSPY=True, colname="Adj Close")
        s_prices = prices[symbol]  # get the price of the entered symbol
        df_trades = s_prices.copy()
        w=20
        BB_ind= id.BBP(symbol, sd, ed, w=20)
        STO_ind= id.STO(symbol, sd, ed, w=20)
        RSI_ind=id.RSI(symbol, sd, ed, w=20)

        C_BB=BB_ind.copy()
        C_STO=STO_ind.copy()
        C_RSI=RSI_ind.copy()
        C_BB[BB_ind > 100]=-1 # sell signal
        C_BB[BB_ind < 0]=1 # buy signal
        C_BB[(BB_ind <= 100) & (BB_ind >=0)]=0
        C_STO[STO_ind > 60] = -1  # sell signal
        C_STO[STO_ind < 20] = 1  # buy signal
        C_STO[(STO_ind <= 70) & (STO_ind >= 20)] = 0  # do nothing

        C_RSI[RSI_ind > 60] = -1  # sell signal
        C_RSI[RSI_ind < 30] = 1  # buy signal
        C_RSI[(RSI_ind <= 60)  & (RSI_ind >= 30)] = 0  # do nothing

        voting_array = C_BB +C_STO +C_RSI
        voting_array[0:w]=0
        pos=0
        T=0
        for i, j in enumerate(voting_array):
            if j==3:
                T=1000-pos
            elif j ==-3:
                T=-1000-pos

            else:
                T=-pos
            df_trades[i] = T
            pos += T
        return df_trades.to_frame()


### Comparing the ManualStrategy and the benchmark


def stats():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    sv = 100000

    # In_sample data
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    impact = 0.005
    orders_df = ManualStrategy.testPolicy('JPM', sd=sd, ed=ed, sv=100000)
    Manual_portvals = compute_portvals(orders_df, sv, commission=9.95, impact=0.005)
    Manual_portvals = Manual_portvals / Manual_portvals.iloc[0]

    dates = pd.date_range(sd, ed)
    prices = get_data(["JPM"], dates, addSPY=True, colname="Adj Close")
    JPM_Prices = prices["JPM"]
    JPM_hold_portvals = JPM_Prices * 1000 + (sv - JPM_Prices[0] * (1 + impact) * 1000) - 9.95
    JPM_hold_portvals = JPM_hold_portvals / JPM_hold_portvals.iloc[0]  # normalized values
    dr = JPM_Prices.copy()
    dr_MS = Manual_portvals.copy()

    # Stats of Portfolio and Benchmark JPM
    dr.iloc[0] = 0
    dr.iloc[1:] = (JPM_hold_portvals.iloc[1:] / JPM_hold_portvals.iloc[:-1].values) - 1  # daily return

    cum_ret_JPM = (JPM_hold_portvals.iloc[-1] / JPM_hold_portvals.iloc[0]) - 1
    avg_daily_ret_JPM = dr.mean()  # mean of dail return
    std_daily_ret_JPM = dr.std()  # std of daily return

    dr_MS.iloc[0] = 0
    dr_MS.iloc[1:] = (Manual_portvals.iloc[1:] / Manual_portvals.iloc[:-1].values) - 1
    cum_ret_MS = (Manual_portvals.iloc[-1] / Manual_portvals.iloc[0]) - 1
    avg_daily_ret_MS = dr_MS.mean()
    std_daily_ret_MS = dr_MS.std()

    # Compare portfolio against JPM benchmark for in-sample
    print(f"Date Range: {sd} to {ed}")
    print()
    print(f"Cumulative Return of Portfolio: {cum_ret_MS}")
    print(f"Cumulative Return of JPM Benchmark : {cum_ret_JPM}")
    print()
    print(f"Standard Deviation of Portfolio: {std_daily_ret_MS}")
    print(f"Standard Deviation of JPM Benchmark : {std_daily_ret_JPM}")
    print()
    print(f"Average Daily Return of Portfolio: {avg_daily_ret_MS}")
    print(f"Average Daily Return of JPM Benchmark : {avg_daily_ret_JPM}")

    # Plotting Manual strategy and benchmark as well as orders lines


    fig = plt.figure()
    plt.plot(JPM_hold_portvals.index, JPM_hold_portvals, color="green")
    plt.plot(Manual_portvals.index, Manual_portvals, color="red")
    # Plotting the orders lines
    for i, v in orders_df.iterrows():
        if v[0] > 0:
            plt.axvline(x=i, color='blue')
        elif v[0] < 0:
            plt.axvline(x=i, color='black')

    plt.title(' In sample ManualStrategy VS Benchmark JPM portfolio value')
    plt.xlabel("Date")
    plt.ylabel("normalized portfolio")
    plt.legend(["Benchmark", "ManualStrategy Portfolio", "Long position", "Short position"])

    fig.autofmt_xdate()
    plt.savefig("In-Sample Benchmark_vs_ManualStrategy")
    plt.clf()

    # Out of sample
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)

    orders_df = ManualStrategy.testPolicy('JPM', sd=sd, ed=ed, sv=100000)
    Manual_portvals = compute_portvals(orders_df, sv, commission=9.95, impact=0.005)
    Manual_portvals = Manual_portvals / Manual_portvals.iloc[0]

    dates = pd.date_range(sd, ed)
    prices = get_data(["JPM"], dates, addSPY=True, colname="Adj Close")
    JPM_Prices = prices["JPM"]
    JPM_hold_portvals = JPM_Prices * 1000 + (sv - JPM_Prices[0] * (1 + impact) * 1000) - 9.95
    JPM_hold_portvals = JPM_hold_portvals / JPM_hold_portvals.iloc[0]  # normalized values
    dr = JPM_Prices.copy()
    dr_MS = Manual_portvals.copy()

    # Stats of Portfolio and Benchmark JPM for out of sample
    dr.iloc[0] = 0
    dr.iloc[1:] = (JPM_hold_portvals.iloc[1:] / JPM_hold_portvals.iloc[:-1].values) - 1  # daily return

    cum_ret_JPM = (JPM_hold_portvals.iloc[-1] / JPM_hold_portvals.iloc[0]) - 1
    avg_daily_ret_JPM = dr.mean()  # mean of dail return
    std_daily_ret_JPM = dr.std()  # std of daily return

    dr_MS.iloc[0] = 0
    dr_MS.iloc[1:] = (Manual_portvals.iloc[1:] / Manual_portvals.iloc[:-1].values) - 1
    cum_ret_MS = (Manual_portvals.iloc[-1] / Manual_portvals.iloc[0]) - 1
    avg_daily_ret_MS = dr_MS.mean()
    std_daily_ret_MS = dr_MS.std()

    # Compare portfolio against JPM benchmark
    print(f"Date Range: {sd} to {ed}")
    print()
    print(f"Cumulative Return of Portfolio: {cum_ret_MS}")
    print(f"Cumulative Return of JPM Benchmark : {cum_ret_JPM}")
    print()
    print(f"Standard Deviation of Portfolio: {std_daily_ret_MS}")
    print(f"Standard Deviation of JPM Benchmark : {std_daily_ret_JPM}")
    print()
    print(f"Average Daily Return of Portfolio: {avg_daily_ret_MS}")
    print(f"Average Daily Return of JPM Benchmark : {avg_daily_ret_JPM}")
    fig = plt.figure()
    plt.plot(JPM_hold_portvals.index, JPM_hold_portvals, color="green")
    plt.plot(Manual_portvals.index, Manual_portvals, color="red")
    for i, v in orders_df.iterrows():
        if v[0] > 0:
            plt.axvline(x=i, color='blue')
        elif v[0] < 0:
            plt.axvline(x=i, color='black')
    plt.title('Out of sample ManualStrategy VS Benchmark JPM portfolio')
    plt.xlabel("Date")
    plt.ylabel("normalized portfolio")
    plt.legend(["Benchmark", "ManualStrategy Portfolio","Long position", "Short position"])

    fig.autofmt_xdate()
    plt.savefig("Out-Sample Benchmark_vs_ManualStrategy")
    plt.clf()

def author():
    return "zelahmadi3"
