
import matplotlib.pyplot as plt
from util import get_data
import pandas as pd
import datetime as dt
from matplotlib.pyplot import figure
def author():
    return "zelahmadi3"

def Volatility(symbol, sd, ed, w=14, gen_plot =False):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, addSPY=True, colname="Adj Close")
    Dates=prices.index # to get the trading days of the stock
    prices=prices[symbol]# selecting the stock column
    vol= prices.rolling(window=w, min_periods=w).std() # calculating standard deviation for window size w
    #normalizing volatility indicator

    VOL= (vol-vol.min())/(vol.max()-vol.min())
    #Plotting the indicator if gen_pot is true
    if gen_plot==True:
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Stock price & Volatility with window size of {}".format(w))
        ax1.plot(Dates, prices,color= "yellow")
        ax1.set(xlabel='Date', ylabel='Stock price')
        ax1.grid()
        ax1.legend()
        ax2.plot(Dates, VOL, color="blue")
        ax2.set(xlabel='Date', ylabel='Volatility')
        ax2.grid()
        ax2.legend(["Volatility"],prop={"size":8})
        fig.autofmt_xdate()
        plt.savefig("Volatility_indicator")
        plt.clf()

    return VOL
#Bollinger Band Percentage Indicator
def BBP(symbol, sd, ed, w=14, gen_plot=False):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, addSPY=True, colname="Adj Close")
    Dates = prices.index  # to get the trading days of the stock
    prices=prices[symbol]
    sma=prices.rolling(window=w,min_periods=w).mean() # calculating average of stock price for window size w
    rolling_std=prices.rolling(window=w, min_periods=w).std()
    UBB=sma+(2*rolling_std) # upper Bollinger Band
    LBB= sma-(2*rolling_std)# Lower Bollinger band
    bbp= 100*(prices-LBB)/(UBB-LBB) # Bollinger Band percentage

    if gen_plot==True:
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(" Stock Price & Bollinger Band% with {} days window".format(w))
        ax1.plot(Dates, prices, color= "yellow")
        ax1.plot(Dates, UBB, color="black")
        ax1.plot(Dates, LBB, color="black")
        ax1.set(xlabel='Date', ylabel='Stock price')
        ax1.grid()
        ax1.legend(["JPM","Bollinger Lower Band","Bolinger Upper Band"],prop={"size":8})

        ax2.plot(Dates, bbp, color="blue")
        ax2.set(xlabel='Date', ylabel='Bolinger% indicator')
        ax2.grid()
        ax2.legend(["BB%"],prop={"size":8})
        fig.autofmt_xdate()
        plt.savefig("BolingerBand%_indicator")
        plt.clf()

    return bbp
# Relative Strength Index
def RSI(symbol, sd, ed, w=14, gen_plot =False):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, addSPY=True, colname="Adj Close")
    prices=prices[symbol]
    Dates = prices.index  # to get the trading days of the stock
    rsi=prices.copy()
    rsi[:]=0 # creating data frame to store rsi

    daily_return = prices.diff()
    daily_return[0]=0
    gains = daily_return.copy() # creating gain data frame
    loss = daily_return.copy() # creating loss data frame
    gains[daily_return < 0] = 0 # making negative values zero
    loss[daily_return >= 0] = 0 #making positive values zero
    G_mean= gains.rolling(window=w, min_periods=w).mean() # calculating the average of gains
    L_mean = abs(loss.rolling(window=w, min_periods=w).mean())# average of losses
    rsi[L_mean==0]=100# rsi =100 when the average losses is zero
    rsi[L_mean!=0]=100-100/(1+(G_mean[L_mean!=0]/L_mean[L_mean!=0]))

    if gen_plot == True:
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Stock Price and RSI with {} days window".format(w))
        ax1.plot(Dates, prices, color="yellow")
        ax1.set(xlabel='Date', ylabel='Stock price')
        ax1.grid()
        ax1.legend()

        ax2.plot(Dates, rsi, color="blue")
        ax2.set(xlabel='Date', ylabel='RSI indicator')
        ax2.grid()
        ax2.legend(["Relative Strength Index"],prop={"size":8})
        fig.autofmt_xdate()
        plt.savefig("RSI_indicator")
        plt.clf()

    return rsi
# Stochastic Oscillator Indicator
def STO(symbol, sd, ed, w=14, gen_plot=False): #Stochastic oscillator
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, addSPY=True, colname="Adj Close")
    prices=prices[symbol]
    Dates = prices.index  # to get the trading days of the stock
    L_w= prices.rolling(window=w, min_periods=w).min() #finding minimum price for window size w
    H_w=prices.rolling(window=w, min_periods=w).max() #finding maximum price for window size w
    ST= 100*(prices-L_w)/(H_w-L_w) # Stochastic Oscillator indicator

    if gen_plot == True:
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Stock Price& Stochastic Oscillator with {} days window".format(w))
        ax1.plot(Dates, prices, color="yellow")
        ax1.set(xlabel='Date', ylabel='Stock_price')
        ax1.grid()
        ax1.legend()

        ax2.plot(Dates, ST, color="blue")
        ax2.set(xlabel='Date', ylabel='Stochastic Oscillator indicator')
        ax2.grid()
        fig.autofmt_xdate()
        plt.savefig("STO_indicator")
        plt.clf()
    return ST
# Momentum Indicator
def MM(symbol, sd, ed, w=14, gen_plot=False):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, addSPY=True, colname="Adj Close")
    prices=prices[symbol]
    Dates = prices.index  # to get the trading days of the stock
    MOM= (prices/prices.shift(w))-1

    if gen_plot == True:
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Stock Price & Momentum with {} days window".format(w))
        ax1.plot(Dates, prices, color="yellow")
        ax1.set(xlabel='Date', ylabel='Stock price')
        ax1.grid()
        ax1.legend()

        ax2.plot(Dates, MOM, color="blue")
        ax2.set(xlabel='Date', ylabel='Momentum indicator')
        ax2.grid()
        ax2.legend(["Momentum"],prop={"size":8})
        fig.autofmt_xdate()
        plt.savefig("MOM_indicator")
        plt.clf()
    return MOM
