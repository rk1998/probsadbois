import pandas as pd
import numpy as np
import os
import scipy as sp
import scipy.stats


def get_csv_data(filename, index=None):
    '''
    Gets data from a csv file and puts it into a pandas dataframe
    '''
    if os.path.exists(filename):
        print( filename + " found ")
        data_frame = pd.read_csv(filename, index_col=index)
        return data_frame
    else:
        print("file not found")



df = get_csv_data("DailyReturn800.csv")

stock_names = df.columns.values

for stock in stock_names:
    currArray = df[stock]
    for i in range(len(currArray)):
        if currArray[i] <= 0:
            currArray[i] = -1
        else:
            currArray[i] = 1
    # for dret in currArray:
    #     if dret <= 0:
    #         dret = -1
    #     else:
    #         dret = 1

print df

# print stock_names
