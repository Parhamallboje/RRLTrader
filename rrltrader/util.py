import pandas as pd
import numpy as np
from pandas_datareader import data
import yaml
import sys
import math 



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