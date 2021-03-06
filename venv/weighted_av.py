from datetime import datetime
from datetime import timedelta
from iexfinance.stocks import get_historical_data
from iexfinance.stocks import get_historical_intraday
import matplotlib.pyplot as plt
import pandas as pd
from main import main
from _200_day_final import f2, create_ticker_data_day
import os
import threading
from ticker_data import ticker_data

def calculate_moving_averages_intra(n, date_list, df):
    y,m,d = date_list
    generic_date = "{}-{}-{}"
    date = generic_date.format(y, m, d)
    sum = 0
    num = 0
    #print(df[date])
    for i in range(n):
        sum += df[date][389-i]
        num += 1
    return float(sum)/num

def weight_av_intra(ticker):
    df = main(ticker)
    moving_average = []
    average_dict = {1: '1 min', 5: '5 min', 10 : '10 min', 15 : '15 min', 20 : '20 min', 25 : '25 min',
                    30: '30 min', 45 : '45 min', 50 : '50 min', 55 : '55 min', 60 : '1 hr',
                    90: '1.5 hr', 120: '2 hr', 150: '2.5 hr', 180: '3 hr', 210: '3.5 hr',
                    240: '4 hr', 270: '4.5 hr', 300: '5 hr', 330: '5.5 hr', 360: '6 hr',
                    390: '6.5 hr'}

    god_data = {}

    good = []
    for col in df[:-1]:
        if 0 not in df[col].values.tolist() and not \
                df[col].isnull().values.any():
            good += [col]

    create_ticker_data_day(ticker)
    for d in good[1:]:
        day_str = d.strftime("%Y-%m-%d")
        final_dict_min = {}
        final_dict_day = {}
        d = day_str.split('-')
        #print('Date: {}'.format(d))
        for i in average_dict:
            temp = calculate_moving_averages_intra(i, d, df)
            moving_average += [temp]
            gen_str = "{} moving average: {}"
            #print(gen_str.format(average_dict[i], temp))
            final_dict_min[average_dict[i]] = temp
        final_dict_day = f2(ticker, d)

        data = final_dict_min
        data.update(final_dict_day)

        god_data[day_str] = data

    final_dict_total = pd.DataFrame.from_dict(god_data)
    directory = './ticker_data'
    file_name = '{}.csv'.format(ticker)
    final_dict_total.to_csv(os.path.join(directory,file_name))


def do_stuff(ticker):
    weight_av_intra(ticker)
    os.remove('tmp_final_{}.csv'.format(ticker))

def loop_stuff(tickers):
    while len(tickers) > 0:
        if len(tickers) > 0:
            t1 = threading.Thread(target=do_stuff, args=(tickers.pop(),))
            t1.start()
        if len(tickers) > 0:
            t2 = threading.Thread(target=do_stuff, args=(tickers.pop(),))
            t2.start()
        if len(tickers) > 0:
            t3 = threading.Thread(target=do_stuff, args=(tickers.pop(),))
            t3.start()
        if len(tickers) > 0:
            t4 = threading.Thread(target=do_stuff, args=(tickers.pop(),))
            t4.start()
        if len(tickers) > 0:
            t5 = threading.Thread(target=do_stuff, args=(tickers.pop(),))
            t5.start()
        if len(tickers) > 0:
            t6 = threading.Thread(target=do_stuff, args=(tickers.pop(),))
            t6.start()
        if len(tickers) > 0:
            t7 = threading.Thread(target=do_stuff, args=(tickers.pop(),))
            t7.start()
        if len(tickers) > 0:
            t8 = threading.Thread(target=do_stuff, args=(tickers.pop(),))
            t8.start()

        t1.join()
        t2.join()
        t3.join()
        t4.join()
        t5.join()
        t6.join()
        t7.join()
        t8.join()

        print("Round Done")


tickers = ticker_data()
ticker = tickers[:12]
loop_stuff(tickers)

print('DONE')

