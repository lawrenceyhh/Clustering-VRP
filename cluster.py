import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import random
from Data.Data_Parser import fetch_all_dataset
from math import radians, cos, sin, asin, sqrt
from sklearn_extra.cluster import KMedoids
from sklearn import metrics, preprocessing, cluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import scipy
from scipy.spatial.distance import pdist, squareform
import plotly.express as px
from kneed import KneeLocator
from scipy.cluster.vq import kmeans2, whiten
from geopy import distance
import sys
import copy
import pickle
import minisom
from kneebow.rotor import Rotor
random.seed(20)


def haversine(lonlat1, lonlat2):
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    r = 6371
    return c * r


def p_median(network, V_c, P, runtime_limit=1800, print_out=False):
    model = gp.Model("formulation of p_median method")

    # supress console output from Gurobi
    if not print_out:
        model.Params.LogToConsole = 0

    # Model Parameters(Stopping Condition)
    model.Params.TimeLimit = runtime_limit
    # model.Params.MIPGap = 3e-2

    # Decision Variables
    x = model.addVars(V_c, vtype=GRB.BINARY, name="Routing")
    y = model.addVars(V_c, V_c, vtype=GRB.BINARY, name="Assignment")
    network_1 = copy.deepcopy(network)
    # variable cost
    VC = 0
    for j in V_c:
        for i in V_c:
            if i != j:
                VC = VC + network_1.nodes[i].demand * \
                     network_1.routes[i, j].total_distance * y[i, j]

    # Objective => minimize fixed cost + variable cost
    model.setObjective(VC, GRB.MINIMIZE)

    # Original constraints without additional bounding constraints:

    # (1) assignment constraint
    model.addConstrs(
        gp.quicksum(y[i, j]
                    for j in V_c) == 1
        for i in V_c
    )

    # (2) link y and x together
    model.addConstrs(
        y[i, j] <= x[j]
        for i in V_c for j in V_c
    )

    # (3) link y and x together
    model.addConstr(
        gp.quicksum(x[j]
                    for j in V_c) == P
    )

    model.optimize()

    # Objective Value:
    objective_value = model.getObjective().getValue()
    if print_out:
        print(f"Objective: {objective_value}")

    # Solution Routes:
    solution = {}
    for j in V_c:
        if round(x[j].X) == 1.0:
            if print_out:
                print(j)
            solution[network_1.nodes[j].coordinate] = []
            for k in V_c:
                if round(y[k, j].X) == 1.0:
                    solution[network_1.nodes[j].coordinate].append(k)

    # Save Runtime for comparison
    runtime = model.Runtime

    # Save MIP Gap for comparison
    mip_gap = model.MIPGap

    model.dispose()
    solution = convert_cluster_centers(network, solution)
    return solution


def kmeans(network, k_cluster, auto=False):
    coordinates = []
    for id in list(network.nodes.keys()):
        if not id.startswith("D"):
            coordinates.append(network.nodes[id].coordinate)
    X = pd.DataFrame(coordinates)
    pass
    if auto:
        max_k = 20
        # iterations
        distortions, score = [], []
        for i in range(2, max_k + 1):
            if len(X) >= i:
                model = KMeans(n_clusters=i, init='k-means++',
                               max_iter=300, random_state=0)
                model.fit(X)
                score.append(metrics.silhouette_score(
                    X, model.labels_, metric='euclidean'))
                distortions.append(model.inertia_)
        # best k: the lowest derivative
        k_cluster = [i * 100 for i in np.diff(distortions, 2)].index(
            min([i * 100 for i in np.diff(distortions, 2)]))
    # plot
    # fig, ax = plt.subplots()
    # ax.plot(range(1, len(distortions) + 1), distortions)
    # ax.axvline(k, ls='--', color="red", label="k = " + str(k))
    # ax.set(title='The Elbow Method', xlabel='Number of clusters',
    #        ylabel="Distortion")
    # ax.legend()
    # ax.grid(True)
    # plt.show()
    model2 = KMeans(n_clusters=k_cluster, init='k-means++',
                    max_iter=300, random_state=0)
    model2.fit(X)
    cluster_center = model2.cluster_centers_.tolist()
    labels = model2.labels_.tolist()
    solution = {}
    for i in range(len(cluster_center)):
        solution[(cluster_center[i][0], cluster_center[i][1])] = []
    for k in range(len(labels)):
        for j in network.nodes:
            if network.nodes[j].coordinate == coordinates[k]:
                solution[(cluster_center[labels[k]][0], cluster_center[labels[k]][1])].append(j)
                break
    solution = recenter_clusters(network, solution)
    solution = convert_cluster_centers(network, solution)
    return solution


def kmedoids(network, k):
    coordinates = []
    for id in list(network.nodes.keys()):
        if not id.startswith("D"):
            coordinates.append(network.nodes[id].coordinate)
    X = pd.DataFrame(coordinates)
    pass
    max_k = 20
    # iterations
    distortions, score = [], []
    # for i in range(2, max_k + 1):
    #     if len(X) >= i:
    #         model = KMedoids(n_clusters=i, init='random', max_iter=300, random_state=0)
    #         model.fit(X)
    #         score.append(metrics.silhouette_score(np.array(X), model.labels_, metric='euclidean'))
    #         distortions.append(model.inertia_)
    model2 = KMedoids(n_clusters=k, init='random',
                      max_iter=300, random_state=0)
    model2.fit(X)
    cluster_center = model2.cluster_centers_.tolist()
    labels = model2.labels_.tolist()
    solution = {}
    for i in range(len(cluster_center)):
        solution[(cluster_center[i][0], cluster_center[i][1])] = []
    for k in range(len(labels)):
        for j in network.nodes:
            if network.nodes[j].coordinate == coordinates[k]:
                solution[(cluster_center[labels[k]][0], cluster_center[labels[k]][1])].append(j)
                break
    solution = convert_cluster_centers(network, solution)
    return solution


def kmeans_DBSCAN(network):
    coordinates = []
    for id in list(network.nodes.keys()):
        if not id.startswith("D"):
            coordinates.append(network.nodes[id].coordinate)
    X = pd.DataFrame(coordinates)
    distance_matrix = squareform(pdist(X, (lambda u, v: haversine(u, v))))
    # find eps
    neigh = NearestNeighbors(n_neighbors=2, metric=lambda u, v: haversine(u, v))
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = pd.DataFrame(distances)
    distances[0] = [*range(0, len(distances[1]), 1)]
    distances = np.array(distances)
    rotor = Rotor()
    rotor.fit_rotate(distances)
    elbow_index = rotor.get_elbow_index()
    best_eps = distances[elbow_index][1]
    db = DBSCAN(eps=best_eps, min_samples=4, metric='precomputed')
    y_db = db.fit_predict(distance_matrix)
    X['cluster'] = y_db
    results = {}
    str_results = {}
    for i in X.values:
        if i[2] not in results.keys():
            results[i[2]] = [[i[1], i[0]]]
            for j in network.nodes:
                if network.nodes[j].coordinate == (i[0], i[1]):
                    str_results[i[2]] = [j]
                    break
        else:
            results[i[2]].append([i[1], i[0]])
            for p in network.nodes:
                if network.nodes[p].coordinate == (i[0], i[1]):
                    str_results[i[2]].append(p)
                    break
            # str_results[i[2]].append('C' + str(count))
    results = dict(sorted(results.items()))
    str_results = dict(sorted(str_results.items()))
    solution = {}
    for k in results.keys():
        xy = np.array(results[k])
        model = KMeans(n_clusters=1, init='k-means++',
                       max_iter=300, random_state=0)
        model.fit(xy)
        solution[(model.cluster_centers_.tolist()[0][1],
                  model.cluster_centers_.tolist()[0][0])] = str_results[k]
    solution = recenter_clusters(network, solution)
    solution = convert_cluster_centers(network, solution)
    return solution


def som_cluster(df, map_shape=(3, 3), num_iteration=100000):
    X = df[df['Id'].apply(lambda x: x.startswith("C"))][["Lat", "Lon"]]
    # scale data
    scaler = preprocessing.StandardScaler()
    X_preprocessed = scaler.fit_transform(X.values)
    # clustering
    model = minisom.MiniSom(x=map_shape[0], y=map_shape[1],
                            input_len=X.shape[1])
    model.train_batch(
        X_preprocessed, num_iteration=num_iteration, verbose=False)
    model.distance_map().T
    df_X = X.copy()
    df_X["cluster"] = np.ravel_multi_index(np.array(
        [model.winner(x) for x in X_preprocessed]).T, dims=map_shape)
    # find real centroids
    cluster_centers = np.array([vec for center in model.get_weights()
                                for vec in center])
    closest, distances = scipy.cluster.vq.vq(cluster_centers, X_preprocessed)
    df_X["centroids"] = 0
    for i in closest:
        df_X["centroids"].iloc[i] = 1
    # add clustering info to the original dataset
    df[["cluster", "centroids"]] = df_X[["cluster", "centroids"]]

    df = df[~df['cluster'].isna()]
    df['cluster'] = df['cluster'].astype(int)
    df['centroids'] = df['centroids'].astype(int)
    solution = {}
    for i in np.sort(df['cluster'].unique()):
        centroid = df[(df['cluster'] == i) & (df['centroids'] == 1)]
        childern = df[(df['cluster'] == i) & (df['centroids'] == 0)]
        # cen_coor = centroid.iloc[0]['Lon'], centroid.iloc[0]['Lat']
        solution[centroid.iloc[0]['Id']] = np.array(childern['Id'])
    return solution


def recenter_clusters(data, cluster_result):
    new_result = {}
    for current_center, children_nodes in cluster_result.items():
        # center initialization
        min_distance = sys.float_info.max
        coor_center = (current_center[1], current_center[0])

        # find min for each cluster
        for node_id in children_nodes:
            current_node = data.nodes.get(node_id)
            # print(node_id)
            coor_node = (
                current_node.coordinate[1], current_node.coordinate[0])
            current_distance = distance.distance(coor_node, coor_center)
            if min_distance > current_distance:
                min_distance = current_distance
                center_node = node_id
        children_nodes.remove(center_node)
        new_result[data.nodes.get(center_node).coordinate] = children_nodes
    return new_result


def convert_cluster_centers(data, cluster_result):
    new_result = {}
    for current_center, children_nodes in cluster_result.items():
        for node_id, node in data.nodes.items():
            if node.coordinate == current_center:
                if node_id in children_nodes:
                    # print("FOUND")
                    children_nodes.remove(node_id)
                new_result[node_id] = children_nodes
                break
    return new_result
