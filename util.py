import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
import yaml
import sys
import math 

plt.style.use('ggplot')


def LoadConfig(
    yamlpath: str)-> dict:
    config = yaml.load(
        open(yamlpath, 'r'), 
        Loader=yaml.FullLoader) 
    return config


def GetData(
    ticker : str,
    start_date : str,
    end_date : str)-> pd.DataFrame:
    """Getting historic price data from yahoo finance.
    
    Arguments:
        ticker {str} 
        start_date {str} 
        end_date {str} 
    
    Returns:
        pd.DataFrame --> the output price dataframe
    """
    return data.DataReader(ticker,'yahoo', start_date, end_date)


def PlotOptimalSharpeRatio(Agent):
    plt.plot(range(len(Agent.epoch_training)),Agent.epoch_training, color ="navy")
    plt.title("Sharpe ratio optimization")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True)
    plt.savefig("outputfiles/graphs/Sharpe ratio optimization {} SMA noFeatures.png".format(str(Agent.input_size)), dpi=300)
    plt.close

def PlotTraining(Agent):
    fig, ax = plt.subplots(nrows=3, figsize=(20, 10))
    t = np.linspace(1, Agent.trading_periods, Agent.trading_periods)[::-1]
    ax[0].plot(t, Agent.prices[:Agent.trading_periods], color="navy")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("SMA-S&P500")
    ax[0].grid(True)

    ax[1].plot(t, Agent.action_space[:Agent.trading_periods], color="navy")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("Trading Signal")
    ax[1].grid(True)

    ax[2].plot(t, Agent.sumR, color="navy", label="Optimised Policy")
    ax[2].plot(range(len(Agent.returns[:Agent.trading_periods])), np.cumsum(Agent.returns[::-1][:Agent.trading_periods]), color="maroon", label="Benchmark")
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("Cumulative Return")
    ax[2].legend(loc="upper left")
    ax[2].grid(True)

    plt.savefig("outputfiles/graphs/rrl_train{}.png".format(str(Agent.input_size)), dpi=300)
    fig.clear()

def PlotWeight(Agent):
    plt.bar(range(len(Agent.w_opt)),Agent.w_opt, color ="navy")
    plt.title("Optimal Weights")
    plt.xlabel("Input Vector Order")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.savefig("outputfiles/graphs/weights{}.png".format(str(Agent.input_size)), dpi=300)
    plt.close
 

def PlotSMA(Agent):
    df = pd.DataFrame(data=Agent.prices[::-1], index=None, columns=["a"])
    price = df["a"]
    rolling_mean = price.rolling(window=5).mean()
    rolling_std = price.rolling(window=5).std() 
    upper_bol = rolling_mean + rolling_std ** 2
    lower_bol = rolling_mean - rolling_std ** 2
    plt.plot(range(len(price)), price, label='S&P500' , color ="navy")
    plt.plot(range(len(price)), rolling_mean, label='50 Day SMA', color='orange')
    plt.plot(range(len(price)), upper_bol, label='50 Upper Bollinger Band', color='green')
    plt.plot(range(len(price)), lower_bol, label='50 Lower Bollinger Band', color='green')
    plt.legend(loc='upper left')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.savefig("outputfiles/SMAExample{}.png".format(str(Agent.input_size)), dpi=300)
    plt.close