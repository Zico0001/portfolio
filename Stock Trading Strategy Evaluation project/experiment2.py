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

def experiment2():
    symbol = 'JPM'
    sv = 100000

    # In_sample data
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    learner= StrategyLearner(verbose = False, impact = 0.0, commission=0.0)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    SL_orders_df=learner.testPolicy(symbol,sd, ed, sv)
    SL_portvals=compute_portvals(SL_orders_df, sv, commission=0.0, impact=0.0)
    SL_portvals = SL_portvals / SL_portvals.iloc[0]

    learner1 = StrategyLearner(verbose=False, impact=0.01, commission=0.0)
    learner1.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    SL_orders_df1 = learner.testPolicy(symbol, sd, ed, sv)
    SL_portvals1 = compute_portvals(SL_orders_df1, sv, commission=0.0, impact=0.01)
    SL_portvals1 = SL_portvals1 / SL_portvals1.iloc[0]

    learner2 = StrategyLearner(verbose=False, impact=0.02, commission=0.0)
    learner2.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    SL_orders_df2 = learner2.testPolicy(symbol, sd, ed, sv)
    SL_portvals2 = compute_portvals(SL_orders_df2, sv, commission=0.0, impact=0.02)
    SL_portvals2 = SL_portvals2 / SL_portvals2.iloc[0]

    dr_sl = SL_portvals.copy()
    dr_sl1 = SL_portvals.copy()
    dr_sl2 = SL_portvals.copy()

    # calculating some Stats ( daily return , cumulative return, etc) for each impact value
    dr_sl.iloc[0] = 0
    dr_sl.iloc[1:] = (SL_portvals.iloc[1:] / SL_portvals.iloc[:-1].values) - 1
    cum_ret_sl = (SL_portvals.iloc[-1] / SL_portvals.iloc[0]) - 1
    avg_daily_ret_sl = dr_sl.mean()
    std_daily_ret_sl = dr_sl.std()

    dr_sl1.iloc[0] = 0
    dr_sl1.iloc[1:] = (SL_portvals1.iloc[1:] / SL_portvals1.iloc[:-1].values) - 1
    cum_ret_sl1 = (SL_portvals1.iloc[-1] / SL_portvals1.iloc[0]) - 1
    avg_daily_ret_sl1 = dr_sl1.mean()
    std_daily_ret_sl1 = dr_sl1.std()

    dr_sl2.iloc[0] = 0
    dr_sl2.iloc[1:] = (SL_portvals2.iloc[1:] / SL_portvals2.iloc[:-1].values) - 1
    cum_ret_sl2 = (SL_portvals2.iloc[-1] / SL_portvals2.iloc[0]) - 1
    avg_daily_ret_sl2 = dr_sl2.mean()
    std_daily_ret_sl2 = dr_sl2.std()
    # Comparing and printing StrategyLearner portfolios with different impact values

    print(f"Date Range: {sd} to {ed}")
    print()
    print(f"Cumulative Return with impact=0: {cum_ret_sl}")
    print(f"Cumulative Return with impact =0.01: {cum_ret_sl1}")
    print(f"Cumulative Return with impact=0.02: {cum_ret_sl2}")
    print()
    print(f"Standard Deviation with impact=0: {std_daily_ret_sl}")
    print(f"Standard Deviation with impact=0.01: {std_daily_ret_sl1}")
    print(f"Standard Deviation with impact=0.02: {std_daily_ret_sl2}")
    print()
    print(f"Average Daily Return with impact=0: {avg_daily_ret_sl}")
    print(f"Average Daily Return  with impact=0.01: {avg_daily_ret_sl1}")
    print(f"Average Daily Return  with impact=0.02: {avg_daily_ret_sl2}")
    fig = plt.figure()
    plt.plot(SL_portvals.index, SL_portvals, color="green")
    plt.plot(SL_portvals1.index, SL_portvals1, color="red")
    plt.plot(SL_portvals2.index, SL_portvals2, color="blue")
    plt.title('StrategyLearner with different impact values')
    plt.xlabel("Date")
    plt.ylabel("normalized portfolio")
    plt.legend(["impact=0.00", "impact=0.01", "impact=0.02"])

    fig.autofmt_xdate()
    plt.savefig("StrategyLearner with different impact values")
    plt.clf()

def author():
    return "zelahmadi3"
if __name__ == "__main__":
    experiment2()
