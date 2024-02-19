import indicators as id
import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
def experiment1():
    symbol='JPM'
    sv = 100000
    impact= 0.005
    commission=9.95
    # In_sample data
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    Manual_orders_df = ManualStrategy.testPolicy(symbol,sd,ed,sv)
    Manual_portvals = compute_portvals(Manual_orders_df,sv, commission=commission, impact=impact)
    Manual_portvals = Manual_portvals / Manual_portvals.iloc[0]


    learner= StrategyLearner(verbose = False, impact = impact, commission=commission)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    SL_orders_df=learner.testPolicy(symbol,sd, ed, sv)
    SL_portvals=compute_portvals(SL_orders_df, sv, commission=commission, impact=impact)
    SL_portvals = SL_portvals / SL_portvals.iloc[0]


    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, addSPY=True, colname="Adj Close")
    JPM_Prices = prices["JPM"]
    JPM_hold_portvals = JPM_Prices * 1000 + (sv - JPM_Prices[0]*(1+impact) * 1000)-commission
    JPM_hold_portvals = JPM_hold_portvals / JPM_hold_portvals.iloc[0]  # normalized values

    dr = JPM_Prices.copy()
    dr_manual = Manual_portvals.copy()
    dr_sl = SL_portvals.copy()

    # Stats of Portfolio and Benchmark JPM
    dr.iloc[0] = 0
    dr.iloc[1:] = (JPM_hold_portvals.iloc[1:] / JPM_hold_portvals.iloc[:-1].values) - 1  # daily return

    cum_ret_JPM = (JPM_hold_portvals.iloc[-1] / JPM_hold_portvals.iloc[0]) - 1
    avg_daily_ret_JPM = dr.mean()  # mean of dail return
    std_daily_ret_JPM = dr.std()  # std of daily return

    dr_manual.iloc[0] = 0
    dr_manual.iloc[1:] = (Manual_portvals.iloc[1:] / Manual_portvals.iloc[:-1].values) - 1


    cum_ret_manual = (Manual_portvals.iloc[-1] / Manual_portvals.iloc[0]) - 1
    avg_daily_ret_manual = dr_manual.mean()
    std_daily_ret_manual = dr_manual.std()

    dr_sl.iloc[0] = 0
    dr_sl.iloc[1:] = (SL_portvals.iloc[1:] / SL_portvals.iloc[:-1].values) - 1
    cum_ret_sl = (SL_portvals.iloc[-1] / SL_portvals.iloc[0]) - 1
    avg_daily_ret_sl = dr_sl.mean()
    std_daily_ret_sl = dr_sl.std()
    # Compare portfolio against JPM benchmark
    print(f"Date Range: {sd} to {ed}")
    print()
    print(f"Cumulative Return of ManualStrategy: {cum_ret_manual}")
    print(f"Cumulative Return of StrategyLearner : {cum_ret_sl}")
    print(f"Cumulative Return of JPM Benchmark : {cum_ret_JPM}")
    print()
    print(f"Standard Deviation of ManualStrategy: {std_daily_ret_manual}")
    print(f"Standard Deviation of StrategyLearner: {std_daily_ret_sl}")
    print(f"Standard Deviation of JPM Benchmark : {std_daily_ret_JPM}")
    print()
    print(f"Average Daily Return of ManualStrategy: {avg_daily_ret_manual}")
    print(f"Average Daily Return of StrategyLearner: {avg_daily_ret_sl}")
    print(f"Average Daily Return of JPM Benchmark : {avg_daily_ret_JPM}")
    fig = plt.figure()
    plt.plot(JPM_hold_portvals.index, JPM_hold_portvals, color="green")
    plt.plot(Manual_portvals.index, Manual_portvals, color="red")
    plt.plot(SL_portvals.index, SL_portvals, color="blue")
    plt.title('In sample ManualStrategy VS Benchmark JPM VS StrategyLearner')
    plt.xlabel("Date")
    plt.ylabel("normalized portfolios")
    plt.legend(["Benchmark", "ManualStrategy","StrategyLearner"])

    fig.autofmt_xdate()
    plt.savefig("In_sample Benchmark_vs_ManualStrategy_vs StrategyLearner")
    plt.clf()
def author():
    return "zelahmadi3"

if __name__ == "__main__":
    experiment1()
