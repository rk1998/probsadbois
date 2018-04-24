# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????
# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????



# CLUSTER
from sklearn.covariance import empirical_covariance
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# WAIT NEVERMIND, IGNORE THIS FILE, DAILYRETURN.PY IS ALREADY STITCHED????

# EXPON, CONFIDENCE
import pandas as pd
import numpy as np
import os
import scipy as sp
import scipy.stats


# CLUSTERING FINDING
# CLUSTERING FINDING
# CLUSTERING FINDING
# CLUSTERING FINDING
def clustering(data, column_names):
    clustering_scores = []
    best_cluster = 2
    best_score = -100
    for i in range(2, 30):
        em = GaussianMixture(n_components=i, covariance_type='tied',
            max_iter=500)
        em.fit(data)
        score = silhouette_score(data, em.predict(data))
        if score > best_score:
            best_score = score
            best_cluster = i
        clustering_scores.append(score)
    em = GaussianMixture(n_components=22, covariance_type='tied', max_iter=500)
    em.fit(data)
    labels = em.predict(data)
    covariance = em.covariances_
    stock_to_cluster_map = {}
    for i in range(0, len(labels)):
        stock_to_cluster_map[column_names[i]] = labels[i]
    return clustering_scores, range(2,30), stock_to_cluster_map, labels, covariance

def plot_results(x_val, y_val, title, x_label, y_label):
    plt.figure(1)
    plt.plot(x_val, y_val, 'o-', color='b', label='Silhouette Score')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


#EXPONENTIAL Distribution
#EXPONENTIAL
#EXPONENTIAL
#EXPONENTIAL
#EXPONENTIAL
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

# Finding the pos and neg values of dret
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

# Finding the x values for an individual stock
want_cdf_of_stock = 'AAME'
count_x = 0
i = 1
while list(df[want_cdf_of_stock])[-i] <= 0:
    i += 1
    count_x += 1

# CDF of Exponential Distribution
l = MLEdict['AAME']
cdf = 1 - np.exp(-l*count_x)
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
