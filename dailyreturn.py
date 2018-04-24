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


def plot_return_curve(stock_data, stock_names, investment_period, initial_investment=1000):
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
            print(closing_value)
            money_made += closing_value - starting_values[j]
            starting_values[j] = closing_value
        result_per_day.append(money_made)
    plt.figure(1)
    plt.plot(range(0, investment_period), result_per_day, '-', color='r', label='Assets')
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



def exponential_distribution():
    '''
    '''
    print("b")

df = get_csv_data("DailyReturn800.csv")
stock_names = df.columns
training_data, testing_data = train_test_split(df.values, test_size=0.3, train_size=0.7, shuffle=False)
training_data_frame = pd.DataFrame(data=training_data, index=None, columns=stock_names)
testing_data_frame = pd.DataFrame(data=testing_data, index=None, columns=stock_names)
plot_return_curve(testing_data_frame, stock_names, 600, initial_investment=100)
scores, cluster_range, clusterlist, labels, covariance_matrix, = clustering(np.transpose(training_data), stock_names.values)
#print(labels)

#puts clusters into 2D array
clusters = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for stock in clusterlist:
    clusters[clusterlist[stock]].append(stock)
#print(clusters)

#gets top stock per cluster using kurtosis
top_from_clusters = []
for cluster in clusters:
    maxindex = 0
    for i in range(len(cluster)):
        if scipy.stats.kurtosis(training_data_frame[cluster[i]]) > scipy.stats.kurtosis(training_data_frame[cluster[maxindex]]):
            maxindex = i
    top_from_clusters.append(cluster[maxindex])
#print(top_from_clusters)

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

#plot_results(cluster_range, scores,
   # "Expectation Maximization Clustering", "Number of Clusters", "Score")
