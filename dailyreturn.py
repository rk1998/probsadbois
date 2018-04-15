import pandas as pd
import numpy as np
import os
from sklearn.covariance import empirical_covariance
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
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

def clustering(data):
    clustering_scores = []
    best_cluster = 2
    best_score = -100
    for i in range(2, 30):
        em = GaussianMixture(n_components=i, covariance_type='tied',
            max_iter=1000)
        em.fit(data)
        score = silhouette_score(data, em.predict(data))
        if score > best_score:
            best_score = score
            best_cluster = i
        clustering_scores.append(score)

    em = GaussianMixture(n_components=2, covariance_type='tied')
    em.fit(data)
    labels = em.predict(data)
    covariance = em.covariances_
    return clustering_scores, range(2,30), labels, covariance

def plot_results(x_val, y_val, title, x_label, y_label):
    plt.figure(1)
    plt.plot(x_val, y_val, 'o-', color='b', label='Silhouette Score')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


df = get_csv_data("DailyReturn800.csv")
scores, cluster_range, clusters, covariance_matrix = clustering(np.transpose(df.values))
print(clusters)
print(covariance_matrix)
plot_results(cluster_range, scores,
    "Expectation Maximization Clustering", "Number of Clusters", "Score")


