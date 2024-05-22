import math
import osmnx as ox
import networkx as nx
import numpy as np
from Data.Data_Parser import Node
from Models import run_F4
from geopy import distance
from tqdm.notebook import tqdm


def preprocess_clusters(network, demands, cluster_result, radius):
    vertices = [v for v in network.nodes.keys() if v.startswith("D")]
    center_lookup = {}

    for index, (center_id, cluster_children) in enumerate(cluster_result.items()):
        station_id = f'P{index}'
        center_lookup[station_id] = center_id
        station_demand = demands[center_id]
        vertices.append(station_id)
        station_coor = network.nodes[center_id].coordinate
        #     station_nnd = ox.nearest_nodes(G, station_coor[0], station_coor[1])

        for child_id in cluster_children:
            node_one = network.nodes.get(child_id)
            # node_one_nnd = ox.nearest_nodes(G, node_one.coordinate[0], node_one.coordinate[1])
            # current_distance = nx.shortest_path_length(G, station_nnd, node_one_nnd, method='bellman-ford', weight="length")
            current_distance = distance.distance(
                (node_one.coordinate[1], node_one.coordinate[0]), 
                (network.nodes[center_id].coordinate[1], network.nodes[center_id].coordinate[0])
            ).km

            # if distance betw/ node and packet station is too large, keep the node
            if current_distance >= radius:
                # arcs[(station_id, child_id)] = current_distance
                # arcs[(child_id, station_id)] = current_distance
                vertices.append(child_id)
                continue

            # update current demand
            station_demand += node_one.demand
            del demands[child_id]
        del demands[center_id]
        # create new station node and update network
        demands[station_id] = station_demand
        station_node = Node(station_id, station_coor, station_demand)
        network.nodes[station_id] = station_node
    return network, vertices, demands, center_lookup


def calculate_arcs(network, vertices, G):
    arcs = {}
    for i in tqdm(vertices):
        for j in vertices:
            if i == j:
                continue
            node_one = network.nodes.get(i)
            node_two = network.nodes.get(j)
            orig_node = ox.nearest_nodes(
                G, node_one.coordinate[0], node_one.coordinate[1])
            dest_node = ox.nearest_nodes(
                G, node_two.coordinate[0], node_two.coordinate[1])
            try:
                current_distance = 0.001 * nx.shortest_path_length(G, orig_node, dest_node, method='bellman-ford',
                                                                   weight='length')
            except nx.NetworkXNoPath:
                current_distance = distance.distance(
                    node_one.coordinate, node_two.coordinate).km
            arcs[(i, j)] = current_distance
    return arcs


def fetch_arcs(network, vertices, center_lookup):
    arcs = {}
    for i in vertices:
        for j in vertices:
            if i == j:
                continue
            if center_lookup.get(i):
                node_one = center_lookup.get(i)
            else:
                node_one = i
            if center_lookup.get(j):
                node_two = center_lookup.get(j)
            else:
                node_two = j
            arcs[(i, j)] = network.routes[(node_one, node_two)].total_distance
    return arcs


def cluster_MILP(data, vertices, arcs, F, alpha, K, Q, demands, cluster_result, radius, runtime_limit=300, gap_acceptance=None, print_out=True, fast_cal=True):
    _network, _vertices, _demands, _center_lookup = preprocess_clusters(data, demands, cluster_result, radius)
    _arcs = fetch_arcs(_network, _vertices, _center_lookup)
    
    least_vehicles_needed = math.ceil(max(_demands.values()) / Q[0])
    
    if fast_cal:
        _Q = np.array(Q) * least_vehicles_needed
        _F = np.array(F) * least_vehicles_needed
        _alpha = np.array(alpha) * least_vehicles_needed
        _K = K
    else:
        _Q = np.array([np.array(Q) * i for i in range(1, least_vehicles_needed + 1)]).reshape(1, -1)[0]
        _F = np.array([np.array(F) * i for i in range(1, least_vehicles_needed + 1)]).reshape(1, -1)[0]
        _alpha = np.array([np.array(alpha) * i for i in range(1, least_vehicles_needed + 1)]).reshape(1, -1)[0]
        _K = list(range(len(_Q)))
    
    V_d = [v for v in _vertices if v.startswith("D")]
    V_c = [v for v in _vertices if v not in V_d]
    
    objective_value, solution, runtime, mip_gap = run_F4(_demands, _vertices, _arcs, V_d, V_c, _F, _alpha, _K, _Q,
                                                         runtime_limit=runtime_limit, gap_acceptance=gap_acceptance, print_out=print_out)
    result_dict = {
        "obj_value": objective_value,
        "runtime": runtime,
        "mip_gap": mip_gap,
        "solution": solution,
        'center': _center_lookup
    }
    return result_dict, _vertices, _demands