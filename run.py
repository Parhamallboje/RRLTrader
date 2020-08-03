
from agents import BaseAgent
from util import LoadConfig, PlotOptimalSharpeRatio, PlotTraining, PlotWeight

    
if __name__ == "__main__":
    dataconfig = LoadConfig('config/DataConfig.yaml')
    rrltraderconfig = LoadConfig('config/rrltraderconfig.yaml')

    InitialAgent = BaseAgent.BaseRRLTrader(
        trading_periods = rrltraderconfig["trading_periods"], 
        start_period = rrltraderconfig["start_period"],
        learning_rate = rrltraderconfig["learning_rate"],
        n_epochs = rrltraderconfig["n_epochs"],
        transaction_costs = rrltraderconfig["transaction_costs"],
        input_size = rrltraderconfig["input_size"],
        added_features = True)

    InitialAgent.upload_data(
        ticker = dataconfig['ticker'],
        start_date = dataconfig['start_date'],
        end_date = dataconfig['end_date'],
        csv_path="sourcefiles/^GSPC.csv")
    
    
    #InitialAgent.load_weight(epoch_path=rrltraderconfig['weight_path'])
    
    #InitialAgent.simulate_trading_data(method="AFFT")
    #InitialAgent.set_action_space()
    #InitialAgent.calculate_action_returns()
    #InitialAgent.calculate_cumulative_action_returns()
    #InitialAgent.RewardFunction()
    InitialAgent.fit()
    PlotOptimalSharpeRatio(InitialAgent)
    PlotTraining(InitialAgent)
    PlotWeight(InitialAgent)
    
    #InitialAgent.save_weight(
        #epoch_path=rrltraderconfig['epoch_path'],
        #weight_path=rrltraderconfig['weight_path'])

    