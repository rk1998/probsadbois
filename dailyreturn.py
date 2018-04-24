import pandas as pd
import numpy as np
import os
from sklearn.covariance import empirical_covariance
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats

#import final_project as fp#


def calculate_return_curve(stock_data, stock_names, investment_period, initial_investment=1000):
    '''
    Calculates the amount of money you make on a set of stocks
    stock_data - dataframe of testing set
    stock_names - stocks you are investing in
    investment_period - number of days you are investing
    initial_investment - amount of money you are investing in each stock
    '''
    result_per_day = []
    starting_values = [initial_investment]*len(stock_names)
    for i in range(0, investment_period):
        money_made = 0
        for j in range(0, len(stock_names)):
            return_values = stock_data[stock_names[j]]
            closing_value = starting_values[j] + (starting_values[j] * (return_values[i]/100.0))
            money_made += closing_value
            starting_values[j] = closing_value
        result_per_day.append(money_made)
    return range(0, investment_period), result_per_day

def plot_return_curve(investment_period, initial_returns, algo_returns, rejected_returns):
    '''
    Plots the return curve
    '''
    plt.figure(1)
    plt.plot(investment_period, initial_returns, '-', color='r', label='Initial Returns')
    plt.plot(investment_period, algo_returns, '-', color='b', label='Our Algo')
    plt.plot(investment_period, rejected_returns, '-', color='g', label='Rejected Stocks')
    plt.title("Assets Over Time")
    plt.xlabel("Time (Days)")
    plt.ylabel("USD")
    plt.legend()
    plt.show()

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

def clustering(data, column_names):
    '''
    Clusters Training Data to create groups of stocks that are
    similar to each other based on covariance.
    '''
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
    '''
    Plots Silhouette Scores of Clustering Experiments
    '''
    plt.figure(1)
    plt.plot(x_val, y_val, 'o-', color='b', label='Silhouette Score')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def mean_confidence_interval(data, confidence=0.95):
    '''
    Finds the mean confidence Interval on a Stock's Daily Return Values
    '''
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def exponential_distribution(data_frame, stock_names):
    '''
    Fits the stock's daily returns to the CDF of an exponential distribution
    '''
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
        cdfs[stock] = 1 - np.exp(-l*count_x)
    return cdfs

df = get_csv_data("DailyReturn800.csv")
stock_names = df.columns
training_data, testing_data = train_test_split(df.values, test_size=0.3, train_size=0.7, shuffle=False)
training_data_frame = pd.DataFrame(data=training_data, index=None, columns=stock_names)
testing_data_frame = pd.DataFrame(data=testing_data, index=None, columns=stock_names)
investment_period, return_values = calculate_return_curve(testing_data_frame, stock_names, 600, initial_investment=1.25)
scores, cluster_range, clusterlist, labels, covariance_matrix, = clustering(np.transpose(training_data), stock_names.values)
#print(labels)

#puts clusters into 2D array
clusters = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for stock in clusterlist:
    clusters[clusterlist[stock]].append(stock)
#print(clusters)

#gets top stock per cluster using kurtosis and mean
top_from_clusters = []
for cluster in clusters:
    top_kurt = {}
    top_mean = {}
    for i in range(len(cluster)):
        top_kurt[scipy.stats.kurtosis(training_data_frame[cluster[i]])] = cluster[i]
        top_mean[mean_confidence_interval(training_data_frame[cluster[i]])[0]] = cluster[i]
    keylist1 = sorted(top_kurt.keys(), reverse = True)
    sort_kurt = []
    for key in keylist1:
        sort_kurt.append(top_kurt[key])
    keylist2 = sorted(top_mean.keys(), reverse = True)
    sort_mean = []
    for key in keylist2:
        sort_mean.append(top_mean[key])
    for i in range(len(sort_kurt)):
        if sort_mean.index(sort_kurt[i]) <= i:
            top_from_clusters.append(sort_kurt[i])
            break

#create dict of confidence intervals and stock tags
top_ci = {}
for i in top_from_clusters:
    top_ci[mean_confidence_interval(training_data_frame[i])[1]] = i

#sort that data structure and create a list of the top 15 by CI
keylist = sorted(top_ci.keys(), reverse = True)
top_15 = []
for key in keylist:
    top_15.append(top_ci[key])
top_15 = top_15[0:15]
print(top_15)

cdf_map = exponential_distribution(training_data_frame, top_15)
cdf_keylist = sorted(cdf_map.keys(), reverse = True)
final_portfolio = cdf_keylist[0:10]
print(final_portfolio)
rejected_stock_names = list(set(stock_names) - set(final_portfolio))
investment_period, algo_returns = calculate_return_curve(testing_data_frame, final_portfolio, 600, initial_investment=100)
investment_period, rejected_stock_returns = calculate_return_curve(testing_data_frame, rejected_stock_names, 600, initial_investment=1.26)
plot_return_curve(investment_period, return_values, algo_returns, rejected_stock_returns)




#plot_results(cluster_range, scores,
   # "Expectation Maximization Clustering", "Number of Clusters", "Score")
