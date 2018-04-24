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

# Counting the number of consecutive negative daily returns
waitingDict = {}
for stock in stock_names:
    currArray = df[stock]
    waitingDict[stock] = []
    currentnegcount = 0
    for i in range(len(currArray)):
        if currArray[i] <= 0:
            currentnegcount += 1
            currArray[i] = -1
        else:
            if currentnegcount > 0:
                waitingDict[stock] = waitingDict[stock] + [currentnegcount]
            currentnegcount = 0
            currArray[i] = 1



# Finding MLE for lambda for every stock
MLEdict = {}
for stock in stock_names:
    lamb = float(len(waitingDict[stock])) / sum(waitingDict[stock])
    MLEdict[stock] = lamb



# Finding the cdf probability
cdfs = {}
for stock in stock_names:
    count_x = 0
    i = 1
    while list(df[stock])[-i] <= 0:
        i += 1
        count_x += 1
    l = MLEdict[stock]
    cdf[stock] = 1 - np.exp(-l*count_x)


# cdfs is a dict that outputs all the probabilities


# # Finding the x values for an individual stock, the number of most recent
# # days where there has been a negative daily return
# want_cdf_of_stock = 'AAME'
# count_x = 0
# i = 1
# while list(df[want_cdf_of_stock])[-i] <= 0:
#     i += 1
#     count_x += 1
#
# # fitting stock to CDF of Exponential Distribution, this is currently only for # one stock, 'AAME' but can be easily generalized to an array of stocks
# l = MLEdict['AAME']
# # probability of there being a
# cdf = 1 - np.exp(-l*count_x)

print count_x
print MLEdict['AAME']
print cdf













# for stock in stock_names:
#     onezeros = df[stock]
#     currentnegcount = 0
#     for i in range(len(currArray)):
#         if
    # for dret in currArray:
    #     if dret <= 0:
    #         dret = -1
    #     else:
    #         dret = 1





# print waitingDict['A']

# print stock_names
