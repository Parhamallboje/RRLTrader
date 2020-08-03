import time
import pickle
import numpy as np
import pandas as pd
import yaml
from datetime import datetime as dt
import matplotlib.pyplot as plt
from ipython_genutils.py3compat import xrange
from statsmodels.tsa.api import AR
from pandas_datareader import data
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S')



class BaseRRLTrader():
    def __init__(
        self,
        trading_periods : int,
        start_period : int,
        learning_rate : float,
        n_epochs : int,
        transaction_costs : float,
        input_size : int,
        kind = "SR"):
        self.trading_periods = trading_periods
        self.lookback = input_size
        self.input_size = input_size +13
        self.start_date_trading = start_period
        self.learning_rate = learning_rate 
        self.n_epochs = n_epochs
        self.transaction_costs = transaction_costs
        self.all_t = None
        self.all_p = None
        self.time = None 
        self.price = None
        self.returns = None
        self.kind = kind
        self.input_vector = np.zeros([trading_periods, self.input_size+2])
        self.action_space = np.zeros(trading_periods+1)
        self.DSR = np.zeros(trading_periods)
        self.dDSRdR = np.zeros(trading_periods)
        self.dRdF = np.zeros(trading_periods)
        self.dRdFp = np.zeros(trading_periods)
        self.dDSRdw = np.zeros([trading_periods, self.input_size+2])
        self.w = np.ones(self.input_size + 2)
        self.w_opt = np.ones(self.input_size + 2)
        self.epoch_training = np.empty(0)
        self.simulation = False

    def add_features(self):
        prices = np.array(list(self.input_df["Adj Close"])[::-1])
        returns = -np.diff(prices)
        
        self.sma5 = movingaverage(returns, 5)
        self.upper_bol_5 = self.sma5 + self.sma5 ** 2
        self.lower_bol_5 = self.sma5 - self.sma5 ** 2
        
        self.sma10 = movingaverage(returns, 10)
        self.upper_bol_10 = self.sma10 + self.sma10 ** 2
        self.lower_bol_10 = self.sma10 - self.sma10 ** 2
        
        self.sma25 = movingaverage(returns, 25)
        self.upper_bol_25 = self.sma25 + self.sma25 ** 2
        self.lower_bol_25 = self.sma25 - self.sma25 ** 2

        self.sma50 = movingaverage(returns, 50)
        self.upper_bol_50 = self.sma50 + self.sma50 ** 2
        self.lower_bol_50 = self.sma50 - self.sma50 ** 2

        self.MACD_50_25 = self.sma25[:-(len(self.sma25)-len(self.sma50))] - self.sma50
        self.MACD_25_10 = self.sma10[:-(len(self.sma10)-len(self.sma25))] - self.sma25

    
    def upload_data(
        self,
        ticker : str,
        start_date : str,
        end_date : str,
        csv_path = None,
        fetch_data = False):
            """Fetches Data by Ticker &
            saves time, prices and returns a numpy arrays
            within the Agent object.
            """
            if csv_path == None and fetch_data == False:
                raise ValueError("You have to either specify a path or allow fetching ")

            if csv_path == None:
                csv_path = "sourcefiles/{}.csv".format(ticker)
            
            if fetch_data == True:
                df = data.DataReader(
                name=ticker,
                data_source='yahoo', 
                start = start_date, 
                end = end_date)
                df.to_csv(csv_path)

            self.input_df = pd.read_csv(csv_path,index_col="Date")
            self.add_features()
        
            if csv_path == None and fetch_data == False:
                raise ValueError("You have to either specify a path or allow pandas DataReader to fetch data")
            

            all_t = np.array(list(self.input_df.index)[::-1])
            all_p = np.array(list(self.input_df["Adj Close"])[::-1])

            if len(all_t) != len(all_p):
                raise IndexError("Lengths of timeframe array {} and price array {} do not match".format(
                        str(len(all_t)),
                        str(len(all_p))))

            else:
                self.time = all_t[
                    self.start_date_trading:
                    self.start_date_trading +
                    self.trading_periods + self.lookback+1]

                self.prices = all_p[
                    self.start_date_trading:
                    self.start_date_trading+
                    self.trading_periods + self.lookback+1]
              
                self.returns = -np.diff(self.prices)


    def simulate_trading_data(
        self,
        method = "AFFT"):
        if method == "GBM":
            simulation_result = GBM(
                self.prices[0], 
                self.mu, 
                self.sigma, 
                self.trading_periods)
            self.simulated_returns= simulation_result[0]
            self.simulated_intervals = simulation_result[1]
            self.simulation = True
        elif method == "EM":
            simulation_result = EM(
                self.prices[0], 
                self.mu, 
                self.sigma, 
                self.trading_periods)
            self.simulated_returns= simulation_result[0]
            self.simulated_intervals = simulation_result[1]
            self.simulation = True
        elif method == "AR":
            simulation_result = Sim_AR(self.prices)
            self.simulation = True
        elif method == 'AFFT':
            self.simulated_df= AAFT(self.input_df)
            self.simulated_returns = np.array(list(self.simulated_df["Adj Close"].pct_change()))
            self.simulated_returns[0] = 0
            self.simulation = True

        else:
            raise ValueError("The method '{}' is invalid or not supported, try 'GBM' or 'EM'".format(method))


    def set_action_space(self):
        for i in range(self.trading_periods-1, -1 , -1):
            self.feature = False
            self.input_vector[i] = np.zeros(self.input_size+2)
            self.input_vector[i][0] = 1.0
            self.input_vector[i][self.input_size+1] = self.action_space[i+1]
            for j in range(1, self.input_size+1, 1):
                if j > self.lookback : 
                    if self.feature is False:

                        self.input_vector[i][j] = self.sma5[i]
                        self.input_vector[i][j+2] = self.lower_bol_5[i]
                        
                        self.input_vector[i][j+1] = self.upper_bol_5[i] 
                
                        self.input_vector[i][j+3] = self.sma10[i]
                        self.input_vector[i][j+4] = self.upper_bol_10[i] 
                        self.input_vector[i][j+5] = self.lower_bol_10[i]
                    
                        self.input_vector[i][j+6] = self.sma25[i]
                        self.input_vector[i][j+7] = self.upper_bol_25[i] 
                        self.input_vector[i][j+8] = self.lower_bol_25[i]

                        self.input_vector[i][j+9] = self.sma50[i]
                        self.input_vector[i][j+10] = self.upper_bol_50[i]
                        self.input_vector[i][j+11] = self.lower_bol_50[i]

                        self.input_vector[i][j+12] = self.MACD_50_25[i]
                        self.input_vector[i][j+13] = self.MACD_25_10[i]
                        self.feature = True

                    else:
                        pass 
                    
                else:
                    if self.simulation:
                        self.input_vector[i][j] = self.simulated_returns[self.lookback+i-j]
                    else:
                        self.input_vector[i][j] = self.returns[self.lookback+i-j]
            self.action_space[i] = np.tanh(np.dot(self.w, self.input_vector[i]))
            #print(self.action_space[i])
            #print(self.action_space[i])

    def calculate_action_returns(self):
        if self.simulation:
            self.action_returns = (
                self.action_space[1:] * self.simulated_returns[:self.trading_periods] - self.transaction_costs * np.abs(-np.diff(self.action_space)))
            
        else:
            self.action_returns = (
                self.action_space[1:] * self.returns[:self.trading_periods] - self.transaction_costs * np.abs(-np.diff(self.action_space)))
            

    def calculate_cumulative_action_returns(self):
        #print(self.action_returns)
        self.sumR = np.cumsum(self.action_returns[::-1])[::-1]
        #print(self.sumR)
        self.sumR2 = np.cumsum((self.action_returns**2)[::-1])[::-1]
        #print(self.sumR2)

    def RewardFunction(self):
        if self.kind == "DSR":
            self.set_action_space()
            self.calculate_action_returns()
            for i in range(1, self.trading_periods,1):
                self.DSR[i] = self.DifferentialSR(decay=0.5,t=i)
                self.dDSRdR[i] = self.calc_dDSRdR(decay=0.5,t=i)
                #print(self.dDSRdR[i])
                self.dRdF[i] = - (self.transaction_costs * np.sign(self.action_space[i]- self.action_space[i+1]))
                #print(self.dRdF[i])
                if self.simulation:
                    self.dRdFp[i] = self.simulated_returns[:self.trading_periods][i] + self.transaction_costs * np.sign(self.action_space[i]- self.action_space[i-1])
                else:
                    self.dRdFp[i] = self.returns[:self.trading_periods][i] + self.transaction_costs * np.sign(self.action_space[i]- self.action_space[i-1])
                #print(self.dRdFp[i])
                self.dFdw[i] = (1-self.action_space[i]**2) * (self.input_vector[i] + self.w[self.input_size+1] * self.dFdw[i-1])
                #print(self.dFdw[i])
                self.dDSRdw[i] = self.dDSRdR[i] * (self.dRdF[i]* self.dFdw[i] + self.dRdFp[i] * self.dFdw[i-1])
                self.RRLGradient_sum += self.dDSRdw[i]

        elif self.kind == "SR":
            self.set_action_space()
            self.calculate_action_returns()
            self.calculate_cumulative_action_returns()
            self.A      =  self.sumR[0] / self.trading_periods
            self.B      =  self.sumR2[0] / self.trading_periods
            self.S      =  self.A / np.sqrt(self.B - self.A**2)
            self.dSdA   =  self.S * (1 + self.S**2) / self.A
            self.dSdB   = -self.S**3 / 2 / self.A**2
            self.dAdR   =  1.0 / self.trading_periods
            self.dBdR   =  2.0 / self.trading_periods * self.action_returns
            self.dRdF   = - self.transaction_costs * np.sign(-np.diff(self.action_space))
            self.dRdFp  =  self.returns[:self.trading_periods] + self.transaction_costs * np.sign(-np.diff(self.action_space))
            self.dFdw = np.zeros(self.input_size+2)
            self.dFpdw= np.zeros(self.input_size+2)
            self.dSdw = np.zeros(self.input_size+2)
            for i in range(self.trading_periods-1, -1, -1):
                if i != self.trading_periods-1:
                    self.dFpdw = self.dFdw.copy()
                self.dFdw  = (1 - self.action_space[i]**2) *  (self.input_vector[i] + self.w[self.input_size+1] * self.dFpdw)
                self.dSdw += (self.dSdA * self.dAdR + self.dSdB * self.dBdR[i]) * (self.dRdF[i] * self.dFdw + self.dRdFp[i] * self.dFpdw)
                #print(self.dSdw)
            
        else:
            raise ValueError("The kind '{}' is invalid or not supported, try 'DSR'".format(kind))


    def ExperienceReplay(self):
        pre_epoch_times = len(self.epoch_training)
        self.optimised_weight = self.w.copy()
        self.RewardFunction()
        logging.info("INFO: Epoch loop start. Intial DSR: {}".format(str(self.DSR[-1])))
        if self.kind == "DSR":
            self.optimised_Reward = self.DSR[-1]
            start_time = time.process_time()
            for i in range(self.n_epochs):
                self.RewardFunction()
                if self.DSR[-1] > self.optimised_Reward:
                    self.optimised_Reward = self.DSR[-1]
                    self.optimised_Reward_ts= self.DSR
                    self.optimised_weight = self.w.copy()
                self.epoch_S = np.append(self.epoch_training,self.DSR[-1])
                self.update_weight()
                #print(self.w)
                if i % 100 == 100-1:
                    epoch_time = time.clock()
                    logging.info("INFO: Simulation Epoch {}/{} | Optimal DSR: {} | Current DSR {} | Elapsed time {} sec.".format(str(i + pre_epoch_times + 1),str(self.n_epochs + pre_epoch_times),str(self.optimised_Reward),str(self.DSR[-1]),str(epoch_time -start_time)))

                if i % 1500 == 1500-1:
                    self.simulate_trading_data()
                    self.w = self.optimised_weight.copy()
                    self.RewardFunction()
                    self.optimised_Reward = self.DSR[-1]
            end_time = time.clock()
            logging.info("INFO: Simulation period is over now. Transfer Learning will commence:")
        self.simulation =False
        self.RewardFunction()
        logging.info("INFO: Epoch loop start. Intial DSR: {}".format(str(self.DSR[-1])))
        if self.kind == "DSR":
            self.optimised_Reward = self.DSR[-1]
            start_time = time.process_time()
            for i in range(self.n_epochs):
                self.RewardFunction()
                if self.DSR[-1] > self.optimised_Reward:
                    self.optimised_Reward = self.DSR[-1]
                    self.optimised_weight = self.w.copy()
                self.epoch_S = np.append(self.epoch_training,self.DSR[-1])
                self.update_weight()
                if i % 100 == 100-1: 
                    epoch_time = time.clock()
                    logging.info("INFO: Real Data Epoch {}/{} | Optimal DSR: {} | Current DSR {} | Elapsed time {} sec.".format(str(i + pre_epoch_times + 1),str(self.n_epochs + pre_epoch_times),str(self.optimised_Reward),str(self.DSR[-1]),str(epoch_time -start_time)))

            end_time = time.clock()            
            self.w = self.optimised_weight.copy()
            self.RewardFunction()
            #print(np.sum(self.action_returns))
            logging.info("INFO: Epoch loop end. Optimal DSR is : {}".format(str(self.optimised_Reward)))

        else:
            raise ValueError("The kind '{}' is invalid or not supported, try 'DSR'".format(kind))



    def fit(self):

        pre_epoch_times = len(self.epoch_training)

        self.RewardFunction()
        print("Epoch loop start. Initial Sharpe ratio : " + str(self.S) + ".")
        self.S_opt = self.S
        
        tic = time.process_time()
        for e_index in range(self.n_epochs):
            self.RewardFunction()
            if self.S > self.S_opt:
                self.S_opt = self.S
                self.w_opt = self.w.copy()
            self.epoch_training = np.append(self.epoch_training, self.S)
            self.update_weight()
            if e_index % 100 == 100-1: 
                toc = time.clock()
                logging.info("INFO: Epoch {}/{} | Optimal SR: {} | Current SR {} | Elapsed time {} sec.".format(str(e_index + pre_epoch_times + 1),str(self.n_epochs + pre_epoch_times),str(self.S_opt),str(self.S),str(toc - tic)))
        toc = time.clock()
        print("Epoch: " + str(e_index + pre_epoch_times + 1) + "/" + str(self.n_epochs + pre_epoch_times) +". Sharpe ratio: " + str(self.S) + ". Elapsed time: " + str(toc-tic) + " sec.")
        self.w = self.w_opt.copy()
        self.RewardFunction()
        print("Epoch loop end. Optimized Sharpe ratio is " + str(self.S_opt) + ".")



    def update_weight(
        self,
        epsilon = 0.5):
        if random.random() > epsilon:
            self.w = self.w_opt
            self.w += self.learning_rate * self.dSdw
        else:
            self.w = np.random.rand(self.input_size+2)

    def save_weight(
        self,
        epoch_path : str,
        weight_path: str):
        pd.DataFrame(self.w).to_csv("{}/weights.csv".format(epoch_path), header=False, index=False)
        pd.DataFrame(self.epoch_training).to_csv("{}/epochs.csv".format(epoch_path), header=False, index=False)
        
    def load_weight(
        self,
        epoch_path : str):
        tmp = pd.read_csv("{}/weights.csv".format(epoch_path), header=None)
        self.w = tmp.T.values[0]

    def DifferentialSR(
        self,
        t:int, 
        decay=0.5)->float:
        if self.a is not None:
            a_p= self.a
            self.a = ema_mean(self.action_returns,decay,t,self.a)
            delta_a = a_p - self.a
        else: 
            self.a = ema_mean(self.action_returns,decay,t)
            delta_a = 0
        if self.b is not None:
            b_p =self.b
            self.b = ema_std(self.action_returns,decay,t, self.b)
            delta_b = b_p - self.b
        else:
            self.b = ema_mean(self.action_returns,decay,t)
            delta_b = 0

        if self.a**2 > self.b:
            return (self.b * delta_a - 0.5 * self.a *delta_b)/  ( - (abs(self.b-self.a**2))**(1.5))    

        else: 
            return (self.b * delta_a - 0.5 * self.a *delta_b)/ ((self.b-self.a**2)**(1.5))

    def calc_dDSRdR(
        self,
        t : int,
        decay=0.5)->float:
        if self.a is not None:
            a_p=self.a
            self.a = ema_mean(self.action_returns,decay,t,self.a)
            delta_a = a_p - self.a
        else: 
            self.a = ema_mean(self.action_returns,decay,t)
            delta_a = 0
        if self.b is not None:
            b_p =self.b
            self.b = ema_std(self.action_returns,decay,t,self.b)
            delta_b = b_p - self.b
        else:
            self.b = ema_mean(self.action_returns,decay,t)
            delta_b = 0
        if self.a**2 > self.b:
            return (self.b - self.a * self.returns[t]) / (- abs((self.b-self.a**2))**(1.5))
        else:
            return (self.b - self.a * self.returns[t]) / ((self.b-self.a **2)**(1.5))

def ema_mean(
returns: np.array,
decay: float,
t: int,
last_ema = 0)-> float :
    if t == 0:
        return returns[t]
    else:
        ema = returns[t+1] + decay*(returns[t] - last_ema)
    return ema

def ema_std(
returns: np.array,
decay: float,
t: int,
last_ema = 0)-> float:
    if t < 0:
        return 0
    elif t==1:
        return returns[t]
    else:
        ema = returns[t+1] + decay*(returns[t]**2- last_ema)
    return ema



def GBM(
    So : float,
    mu : float, 
    sigma: float,
    N : float) -> list:
    """[summary]
    Arguments:
        So {float} -- initial stock price
        mu {float} -- mean of historical daily returns
        sigma {float} -- standard deviation of historical daily returns
        N {float} -- number of time points in prediction the time horizon
    
    Keyword Arguments:
        T {[type]} -- length of the prediction time horizon (default: {1.})
    
    Returns:
        list -- [description]
    """
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

def EM(
    So : float, 
    mu : float,
    sigma : float, 
    N : int,
    M = 1):
    b = Brownian(N)[1]    
    dt = M * (1/N)  # EM step size
    L = N / M
    wi = [So]
    for i in xrange(0,int(L)):
        Winc = np.sum(b[(M*(i-1)+M):(M*i + M)])
        w_i_new = wi[i]+mu*wi[i]*dt+sigma*wi[i]*Winc
        wi.append(w_i_new)
    return wi, dt


def Brownian(
    increments: int,
    seed = np.random.randint(20000)):
    np.random.seed(seed)                         
    time_step = 1./increments
    brownian_increments = np.random.normal(0., 1., int(increments))*np.sqrt(time_step) 
    brownian_path = np.cumsum(brownian_increments)
    return brownian_increments, brownian_path

def Sim_AR(
    prices: np.array):
    """Vector Autoregressive Baseline Generator."""
    # VAR model
    var = AR(prices)
    model_fit = var.fit()
    # make prediction
    return model_fit.predict(len(prices), len(prices))


def AAFT(df, random=np.random.uniform, random_state=None):
    """Amplitude Adjusted Fourier Transform Baseline Generator."""
    # set random seed
    np.random.seed(random_state)
    # Operate on numpy.ndarray
    ts = df.values
    # 2d time-series format
    _ts = ts.reshape(len(ts), -1)
    # Odd number of samples
    if len(_ts) % 2 != 0:
        _ts = _ts[1:, :]
    # Generated time-series
    ts_gen = np.empty_like(_ts)
    for i, tsi in enumerate(_ts.T):
        # Fourier Transaformation (real-valued signal)
        F_tsi = np.fft.rfft(tsi)
        # Randomization of Phase
        rv_phase = np.exp(random(0, np.pi, len(F_tsi)) * 1.0j)
        # Generation of new time-series
        F_tsi_new = F_tsi * rv_phase
        # Inverse Fourier Transformation
        ts_gen[:, i] = np.fft.irfft(F_tsi_new)
    # Create pandas DataFrame
    df_gen = pd.DataFrame(ts_gen, columns=df.columns,
                          index=df.index[-len(ts_gen):])
    return df_gen

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


