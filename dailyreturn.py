import pandas as pd
import numpy as np
import os
from sklearn.covariance import empirical_covariance
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

df = get_csv_data("DailyReturn800.csv")
stock_names = df.columns
training_data, testing_data = train_test_split(df.values, test_size=0.3, train_size=0.7, shuffle=False)
scores, cluster_range, clusterlist, labels, covariance_matrix, = clustering(np.transpose(training_data), stock_names.values)
print(labels)
clusters = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for stock in clusterlist:
    clusters[clusterlist[stock]].append(stock)
print(clusters)

top_from_clusters = []
for cluster in clusters:
    maxindex = 0
    for i in range(len(cluster)):
        if scipy.stats.kurtosis(df[cluster[i]]) > scipy.stats.kurtosis(df[cluster[maxindex]]):
            maxindex = i
    print(maxindex, cluster[maxindex])
    top_from_clusters.append(cluster[maxindex])
print(top_from_clusters)

#plot_results(cluster_range, scores,
   # "Expectation Maximization Clustering", "Number of Clusters", "Score")
