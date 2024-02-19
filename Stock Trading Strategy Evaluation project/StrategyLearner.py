""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		   	 		  		  		    	 		 		   		 		  
import random as rand
  		  	   		   	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		   	 		  		  		    	 		 		   		 		  
import util as ut
import indicators as id
import QLearner as ql
from marketsimcode import compute_portvals
  		  	   		   	 		  		  		    	 		 		   		 		  

class StrategyLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    # constructor  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		   	 		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		   	 		  		  		    	 		 		   		 		  
        self.commission = commission  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		  	   		   	 		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        sv=10000,
    ):

        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # add your code to do learning here
        dates = pd.date_range(sd, ed)
        prices=ut.get_data([symbol], dates, addSPY=True, colname="Adj Close")
        S_price=prices[symbol]
        state=ind_disc_to_state(symbol, sd, ed, w=20)
        actions=[1, 0, -1] # ( BUY, HOLD, SELL)
        self.q=ql.QLearner(1000, 3, 0.5, 0.9, 0.4, 0.9, 0)
        s= state.iloc[0]
        action= self.q.querysetstate(s)
        #T=0 # trade
        pos=0 # position
       # r=0 # reward
        cash=sv
        w=20
        SP=S_price[w:]
        daily_return=(SP / SP.shift(1)) - 1
        trades = SP.copy()

        trades[:]=0

        for i  in range(1,len(SP)):
            s_prime=state.iloc[i]
            if actions[action]==1:
                T=1000-pos
                r=T*(SP[i]-SP[i-1]*(1+self.impact))-self.commission

            elif actions[action]==-1:
                T=-1000-pos
                r=T*(SP[i]*(1-self.impact)-SP[i-1])-self.commission
            else:
                T=-pos
                r=T*(SP[i]-SP[i-1])
            trades[i] = T
            pos += T
            action = self.q.query(s_prime, r)
        df_trades= pd.DataFrame(trades, columns=[symbol])
        port_vals = compute_portvals(df_trades, impact=self.impact, start_val=sv, commission=self.commission)
        return df_trades
        if self.verbose:
            print(port_vals)
        # example usage of the old backward compatible util function  		  	   		   	 		  		  		    	 		 		   		 		  
        syms = [symbol]  		  	   		   	 		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		  	   		   	 		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		   	 		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols  		  	   		   	 		  		  		    	 		 		   		 		  
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		   	 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(prices)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # example use with new colname  		  	   		   	 		  		  		    	 		 		   		 		  
        volume_all = ut.get_data(  		  	   		   	 		  		  		    	 		 		   		 		  
            syms, dates, colname="Volume"  		  	   		   	 		  		  		    	 		 		   		 		  
        )  # automatically adds SPY  		  	   		   	 		  		  		    	 		 		   		 		  
        volume = volume_all[syms]  # only portfolio symbols  		  	   		   	 		  		  		    	 		 		   		 		  
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later  		  	   		   	 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(volume)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		  	   		   	 		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=10000,
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        dates = pd.date_range(sd, ed)
        prices = ut.get_data([symbol], dates, addSPY=True, colname="Adj Close")
        S_price = prices[symbol]
        actions = [1, 0, -1]
        pos=0

        state = ind_disc_to_state(symbol, sd, ed, w=20)
        w = 20
        SP = S_price[w:]
        daily_return = (SP / SP.shift(1)) - 1
        trades = SP.copy()
        trades[:] = 0
        for i in range(1,len(SP)):
            s_prime=state.iloc[i]
            next_action = self.q.querysetstate(s_prime)

            if actions[next_action]==1: # buy
                T = 1000 - pos
                r=T*(SP[i]-SP[i-1]*(1+self.impact))-self.commission

            elif actions[next_action]==-1: # Sell
                T = -1000 - pos
                r=T*(SP[i]*(1-self.impact)-SP[i-1])-self.commission
            else:                         # do nothing/hold
                T = -pos
                r = T * (SP[i]-SP[i-1])

            trades[i] = T
            pos += T
        df_trades = pd.DataFrame(trades, columns=[symbol])
        port_vals = compute_portvals(df_trades, impact=self.impact, start_val=sv, commission=self.commission)

        if self.verbose:
            print(port_vals)
        return df_trades
        # here we build a fake set of trades
        # your code should return the same sort of data  		  	   		   	 		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		  	   		   	 		  		  		    	 		 		   		 		  
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY  		  	   		   	 		  		  		    	 		 		   		 		  
        trades = prices_all[[symbol,]]  # only portfolio symbols  		  	   		   	 		  		  		    	 		 		   		 		  
        trades_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		   	 		  		  		    	 		 		   		 		  
        trades.values[:, :] = 0  # set them all to nothing  		  	   		   	 		  		  		    	 		 		   		 		  
        trades.values[0, :] = 1000  # add a BUY at the start  		  	   		   	 		  		  		    	 		 		   		 		  
        trades.values[40, :] = -1000  # add a SELL  		  	   		   	 		  		  		    	 		 		   		 		  
        trades.values[41, :] = 1000  # add a BUY  		  	   		   	 		  		  		    	 		 		   		 		  
        trades.values[60, :] = -2000  # go short from long  		  	   		   	 		  		  		    	 		 		   		 		  
        trades.values[61, :] = 2000  # go long from short  		  	   		   	 		  		  		    	 		 		   		 		  
        trades.values[-1, :] = -1000  # exit on the last day  		  	   		   	 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(type(trades))  # it better be a DataFrame!  		  	   		   	 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(trades)  		  	   		   	 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(prices_all)  		  	   		   	 		  		  		    	 		 		   		 		  
        return trades



# scaling, discretizing indicators and convert indicators to state to int
def ind_disc_to_state(symbol, sd, ed, w=20):
    BB_ind = id.BBP(symbol, sd, ed, w=20)
    BB = BB_ind[w:]
    # scaling the indicator to be in range between 0 and 1
    BB_scaled = (BB - BB.min()) / (BB.max() - BB.min())
    STO_ind = id.STO(symbol, sd, ed, w=20)
    STO = STO_ind[w:]
    STO_scaled = (STO - STO.min()) / (STO.max() - STO.min())
    RSI_ind = id.RSI(symbol, sd, ed, w=20)
    RSI = RSI_ind[w:]
    RSI_scaled = (RSI - RSI.min()) / (RSI.max() - RSI.min())
    # discretizing the indicators
    D_BB = pd.cut(BB_scaled,
                  bins=[-0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                  labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    D_STO = pd.cut(STO_scaled,
                   bins=[-0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                   labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    D_RSI = pd.cut(RSI_scaled,
                   bins=[-0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                   labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    D_BB = D_BB.to_frame()
    D_STO = D_STO.to_frame()
    D_RSI = D_RSI.to_frame()

    # concatenating the indicators to create a state

    state = D_BB.astype(str) + D_STO.astype(str) + D_RSI.astype(str)
    state = state.astype(int)
    return state
def author():
    return "zelahmadi3"
if __name__ == "__main__":
    print("One does not simply think up a strategy")  		  	   		   	 		  		  		    	 		 		   		 		  
