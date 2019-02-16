from datetime import datetime
from datetime import timedelta
from iexfinance.stocks import get_historical_data
from iexfinance.stocks import get_historical_intraday
import matplotlib.pyplot as plt
import pandas as pd

d1 = pd.read_csv('goog(2018,12,18).csv')
d2 = pd.read_csv('goog(2018,12,19).csv')
d1.set_index('minute', inplace=True)
d2.set_index('minute', inplace=True)
print(d1)
result = pd.concat([d1, d2], sort=False, axis=1)
result.to_csv('result.csv')
