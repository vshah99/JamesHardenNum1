from datetime import datetime
from datetime import timedelta
from iexfinance.stocks import get_historical_data
from iexfinance.stocks import get_historical_intraday
import matplotlib.pyplot as plt
import pandas as pd

def calculate_moving_averages_intra(n: int, y: int, m: int, d: int) -> int:
    df = pd.read_csv('final.csv')
    generic_date = "{}-{}-{}"
    date = generic_date.format(y,m,d)
    sum = 0
    num = 0
    for i in range(n):
        sum += df[date][389-i]
        num += 1
    return float(sum)/num


moving_average = []

average_dict = {1: '1 min', 5: '5 min', 10 : '10 min', 15 : '15 min', 20 : '20 min', 25 : '25 min',
                30: '30 min', 45 : '45 min', 50 : '50 min', 55 : '55 min', 60 : '1 hr',
                90: '1.5 hr', 120: '2 hr', 150: '2.5 hr', 180: '3 hr', 210: '3.5 hr',
                240: '4 hr', 270: '4.5 hr', 300: '5 hr', 330: '5.5 hr', 360: '6 hr',
                390: '6.5 hr'}

for i in average_dict:
    temp = calculate_moving_averages_intra(i, 2018, 12 ,12)
    moving_average += [temp]
    gen_str = "{} moving average: {}"
    print(gen_str.format(average_dict[i], temp))

#plt.plot(moving_average)
#plt.show()
