import pandas as pd
import numpy as np


from iexfinance.stocks import get_historical_data
from datetime import datetime
from datetime import timedelta
from running_avg_200day import calculate_moving_averages_day

def create_ticker_data_day(ticker: str):
    start = datetime(2017, 11, 2)
    end = datetime(2019, 2, 15)

    df = get_historical_data(ticker, start, end, output_format='pandas')
    drop_list = ['open', 'high', 'low', 'volume']
    df.drop(drop_list, axis=1, inplace=True)
    df.to_csv('tmp_final_{}.csv'.format(ticker))

def f2(ticker: str, date: list) -> dict:
    y, m, d = date
    y = int(y)
    m = int(m)
    d = int(d)

    days_dict = {2: '2 day', 5: '5 day', 10: '10 day', 20: '20 day', 30: '30 day', 40: '40 day',
                50: '50 day', 60: '60 day', 70: '70 day', 80: '80 day', 90: '90 day', 100: '100 day',
                125: '125 day', 150: '150 day', 175: '175 day', 200: '200 day'}
    final_dict_day = {}

    for i in days_dict:
        tmp = calculate_moving_averages_day(i, y, m, d, ticker)
        gen_str = "{} moving avg: {}"
        #print(gen_str.format(days_dict[i], tmp))
        final_dict_day[days_dict[i]] = tmp
    return (final_dict_day)
