import pandas as pd
import numpy as np


from iexfinance.stocks import get_historical_data
from datetime import datetime
from datetime import timedelta


# drop_list = ['open', 'high', 'low', 'volume']
#
# start = datetime(2018, 2, 28)
# end = datetime(2018, 12,12)
#
# df = get_historical_data("GOOG", start, end, output_format = 'pandas')
# df.drop(drop_list,axis=1,inplace=True)
# df.to_csv('goog.csv')

def calculate_moving_averages_day(n: int, y: int, m: int, d: int) -> int:
    df = pd.read_csv('tmp_final.csv')
    #df.drop('date',axis =1, inplace =True)
    #df.set_index("date", inplace = True)
    curr=datetime(y,m,d)
    final = df.index[df.date == curr.strftime("%Y-%m-%d")]
    df.drop('date',axis =1, inplace =True)
    #print(df)
    #print(final)
    sum = 0
    num = 0
    for i in range(n):

        sum += df['close'].values[final-i]
        num += 1
    return float(sum)/num
