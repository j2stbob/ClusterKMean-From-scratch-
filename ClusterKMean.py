import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


data = pd.read_csv("Datas/WineQT.csv")
data = data.drop(columns=["Id", "citric acid", "residual sugar",
                          "chlorides", "free sulfur dioxide",
                          "total sulfur dioxide", "density", "pH",
                          "sulphates", "volatile acidity", "quality"])

data = np.array(data)



# evclid distation
def distation(p1: float, p2: float):
    return np.sqrt(np.sum(p1 - p2)**2)

# random point(centroid)
def random_point(k: int) -> list:
    points = []
    for j in range(k):
        point = np.array([random.randint(0, 20) for i in range(2)])
        points.append(point)
    return np.array(points)

# adds centroids
def update_point(cluster) -> list:
    new_point = []
    for cl in cluster:
        new_point.append(np.array([random.randint(0, 20) for i in range(2)]))
    return np.array(new_point)

# find the cluster for train
def clustering(data, points) -> list:
    cluster = [[] for j in range(len(points))]
    for i in data:
        distances = [distation(i, p) for p in points]
        near_point_index = np.argmin(distances)
        cluster[near_point_index].append(i)
    return cluster



# fit model
def train(data: list, point: int):
    tolerance = 1e-4
    points = random_point(point)
    for i in range(1000):
        clusters = clustering(data, points)
        new_point = update_point(clusters)          # add new point
        if np.all((new_point - points) < tolerance):        # error
            break
        points = new_point
    return clusters, points


# -----------------------------------------------------------------------------------------------------------------------
def draw(clusters, points):
    colors = ['#FF0000', '#00FF00', '#0000FF', '#c5f005', '#c5f010']
    for idx, cluster in enumerate(clusters):                            #index for colour -> colour cluster
        for i in cluster:
            plt.scatter(i[0], i[1], c=colors[idx])
    for i in points:
        plt.scatter(i[0], i[1], c='#000000', marker='x')
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------



# main
clusters, points = train(data, 5)
draw(clusters, points)