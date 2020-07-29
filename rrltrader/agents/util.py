import pandas as pd
import numpy as np
from pandas_datareader import data
import yaml
import sys
import math 
sys.setrecursionlimit(20000)


def SharpeRatio(
    returns : np.array) -> float:
    return np.mean(returns, axis=0) / (np.std(returns, axis=0))

def hit_ratio(
    returns : np.array) -> float:
    return np.sum(returns > 0, axis=0) /len(returns)

def featureNormalizer(X):
	m = len(X)
	mean = np.mean(X)
	std = np.std(X)
	X_norm = (X-mean)/std
	return X_norm , mean , std

def getcolumn(
    df: pd.DataFrame,
    column_name: str)-> np.array:
    return df[column_name].to_numpy()


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


def DSR(
    returns: np.array,
    t:int, 
    decay=0.5
    )->float:
    a = ema_mean(returns,decay,t)
    b = ema_std(returns,decay,t)
    delta_a = ema_mean(returns, decay,t-1) - a
    delta_b = ema_mean(returns, decay,t-1) - b
    if a**2 > b:
        return (b * delta_a - 0.5 *a *delta_b)/  ( - (abs(b-a**2))**(1.5))    

    else: 
        delta_a = ema_mean(returns, decay,t-1) - a
        delta_b = ema_mean(returns, decay,t-1) - b
        return (b * delta_a - 0.5 *a *delta_b)/ ((b-a**2)**(1.5))

def dDSRdR(
    returns: np.array,
    t : int,
    decay=0.5)->float:
    a = ema_mean(returns,decay,t-1)
    b = ema_std(returns,decay,t-1)
    if a**2 > b:
        return (b - a * returns[t]) / (- abs((b-a**2))**(1.5))
    else:
        return (b - a * returns[t]) / ((b-a**2)**(1.5))




def ema_mean(
returns: np.array,
decay: float,
t: int)-> float :
    if t < 0:
        return returns[t]
    else:
        ema = returns[t-1] + decay*(
            returns[t]- ema_mean(
                returns,decay,t-1
                )
                )
    return ema

def ema_std(
returns: np.array,
decay: float,
t: int)-> float:
    if t < 0:
        return 0
    elif t==1:
        return returns[t]
    else:
        ema = returns[t-1] + decay*(
            returns[t]**2- ema_mean(
                returns,decay,t-1
                )
                )
    return ema



