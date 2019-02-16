from datetime import datetime
from datetime import timedelta
from iexfinance.stocks import get_historical_data
from iexfinance.stocks import get_historical_intraday
import matplotlib.pyplot as plt
import pandas as pd

def main(ticker: str):
    final = datetime(2019, 2, 15)
    end = datetime(2018, 12, 10)
    drop_list = ['average', 'changeOverTime', 'date', 'high', 'label', 'low', 'close',
                 'marketLow', 'marketHigh', 'marketNotional', 'marketOpen', 'marketVolume',
                 'marketNumberOfTrades', 'numberOfTrades', 'open', 'volume',
                 'marketAverage', 'marketChangeOverTime', 'notional']

    list_of_df = []
    while (end <= final):
        try:
            df = get_historical_intraday(ticker, end, output_format = 'pandas')
            df.drop(drop_list, axis=1, inplace = True)
            df.rename(columns = {'marketClose': end}, inplace = True)
            df.to_csv('temp.csv')
            df = pd.read_csv('temp.csv')
            df.set_index('minute', inplace=True)
            list_of_df += [df]
            end += timedelta(days=1)
            print(end)
        except:
            end += timedelta(days=1)
    result = pd.concat(list_of_df, sort=False, axis=1)
    result.to_csv('final.csv')

