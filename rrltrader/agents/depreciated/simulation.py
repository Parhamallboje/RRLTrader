import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR as _VAR
from pandas_datareader import data
from ipython_genutils.py3compat import xrange
import matplotlib.pyplot as plt

def GBM(
    So : float,
    mu : float, 
    sigma: float,
    N : float,
    T =1.) -> list:
    W = Brownian(N)[0]
    t = np.linspace(0.,1.,int(N)+1)
    S = []
    S.append(So)
    for i in xrange(1,int(N+1)):
        drift = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i-1]
        S_temp = So*np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t

def Brownian(
    increments: int):
    time_step = 1./increments
    brownian_increments = np.random.normal(
        0.,1., 
        int(increments))*np.sqrt(time_step)
    brownian_path = np.cumsum(brownian_increments)
    return brownian_increments, brownian_path

def AAFT(
    df : pd.DataFrame, 
    random = np.random.uniform, 
    random_state=None) -> pd.DataFrame:
    """Amplitude Adjusted Fourier Transform."""
    np.random.seed(random_state)
    ts = df.values
    _ts = ts.reshape(len(ts), -1)
    if len(_ts) % 2 != 0:
        _ts = _ts[1:, :]
    ts_gen = np.empty_like(_ts)
    for i, tsi in enumerate(_ts.T):
        F_tsi = np.fft.rfft(tsi)
        rv_phase = np.exp(random(0, np.pi, len(F_tsi)) * 1.0j)
        F_tsi_new = F_tsi * rv_phase
        ts_gen[:, i] = np.fft.irfft(F_tsi_new)
    df_gen = pd.DataFrame(ts_gen, columns=df.columns,
                          index=df.index[-len(ts_gen):])
    return df_gen
    