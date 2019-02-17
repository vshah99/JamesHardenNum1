import os
import glob
import pandas as pd
import numpy as np
import random


def load_data_wrapper():
    #path = os.path.dirname(os.path.realpath(__file__))
    allFiles = glob.glob(os.path.join('/Users/vedantshah/PycharmProjects/JamesHardenNo1/venv/ticker_data/',"*.csv"))

    dataframes = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=0, header=0)
        dataframes += [df]


    final = pd.concat(dataframes, axis=1, sort=True)
    final = final.transpose()

    temp_increase = (1,0)
    temp_decrease = (0,1)

    new = []

    count_row = final.shape[0]

    training_inputs = [] #len = 2025

    for i in range(count_row):
        row = final.iloc[i]
        start_price = final.iloc[i].loc['1 min']
        final.iloc[i] = row/start_price
        tod_price = final.iloc[i].loc['1 min']
        tom_price = final.iloc[i].loc['Tomorrow']
        if tom_price > tod_price:
            change = 1
        else:
            change = 0

        input_values = row.values
        #print(input_values)
        training_inputs += [(input_values, change)]
        new += [change]

    final['Change'] = new
    #random.shuffle(training_inputs)
    pd_to_csv = pd.DataFrame(final)
    pd_to_csv.to_csv('TEST.csv')

    #print(training_inputs[0][0])
    '''
    training_data = training_inputs[0:1500]
    test_data = training_inputs[1500:1750]
    validation_data = training_inputs[1750:]

    return (training_data, validation_data, test_data)
    '''


load_data_wrapper()
