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
    plt.plot(range(len(Agent.epoch_training)),Agent.epoch_training, color ="blue")
    plt.title("Sharpe ratio optimization")
    plt.xlabel("Epoch times")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True)
    plt.savefig("outputfiles/graphs/Sharpe ratio optimization {} SMA noFeatures.png".format(str(Agent.input_size)), dpi=300)
    plt.close

def PlotTraining(Agent):
    fig, ax = plt.subplots(nrows=3, figsize=(15, 10))
    t = np.linspace(1, Agent.trading_periods, Agent.trading_periods)[::-1]
    ax[0].plot(t, Agent.prices[:Agent.trading_periods], color="blue")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("S&P500")
    ax[0].grid(True)

    ax[1].plot(t, Agent.action_space[:Agent.trading_periods], color="blue", label="With optimized weights")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("Trader Function")
    ax[1].legend(loc="upper left")
    ax[1].grid(True)

    ax[2].plot(t, Agent.sumR, color="blue", label="Optimised Policy")
    ax[2].plot(range(len(Agent.returns[:Agent.trading_periods])), np.cumsum(Agent.returns[::-1][:Agent.trading_periods]), color="red", label="Benchmark")
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("Cumulative Return")
    ax[2].legend(loc="upper left")
    ax[2].grid(True)
    plt.savefig("outputfiles/graphs/rrl_train{} SMA noFeatures.png".format(str(Agent.input_size)), dpi=300)
    fig.clear()

def PlotWeight(Agent):
    plt.bar(range(len(Agent.w_opt)),Agent.w_opt, color ="blue")
    plt.title("Optimal Weights")
    plt.xlabel("Input Vector Order")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.savefig("outputfiles/graphs/weights{} SMA noFeatures.png".format(str(Agent.input_size)), dpi=300)
    plt.close
 
def PlotSMAandPrice(Agent):
    plt.plot(range(len(Agent.prices)),Agent.prices, color ="blue")
    plt.plot(range(len(Agent.self.sma5)),Agent.self.sma5, color ="blue")
    plt.title("Optimal Weights")
    plt.xlabel("Order")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.savefig("outputfiles/graphs/weights.png", dpi=300)
    plt.close
