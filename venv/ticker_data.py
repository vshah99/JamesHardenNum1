import pandas as pd
import mpu

def ticker_data()
    di = pd.read_excel('companylist.xlsx')

    x=di.values.tolist()
    y = []
    for i in range(0,len(x)):
        y += x[i]
    return y



