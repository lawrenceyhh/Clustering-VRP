import osmnx as ox
import pandas as pd
import sys
from geopy import distance
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from tqdm.notebook import tqdm
import networkx as nx


G = ox.graph.graph_from_xml(filepath='./data/graph.osm')


@dataclass
class Node:
    id: str
    coordinate: Tuple[float, float]
    demand: int


@dataclass
class Route:
    id_from: str
    id_to: str
    total_distance: float
    shortest_path: list


@dataclass
class Network:
    name: str
    nodes: Dict[str, Node]
    routes: Dict[Tuple[str, str], Route]


@dataclass
class Truck:
    model: str
    capacity: float
    fix_cost: float
    varible_cost: float


node_data = pd.read_csv(".\\Data\\Paris.nodes", sep=" ")
Id_x, Lon_x, Lat_x, Id_y, Lon_y, Lat_y, current_distance, shortest_route = [], [], [], [], [], [], [], []
dict_nodes = {}
for i in tqdm(node_data.index):
    current_node = Node(node_data.loc[i, "Id"], (node_data.loc[i, "Lon"], node_data.loc[i, "Lat"]),
                        node_data.loc[i, "Demand[kg]"])
    if current_node.id not in dict_nodes.keys():
        dict_nodes[current_node.id] = current_node

dict_routes = {}
for node_one_id, node_one in tqdm(dict_nodes.items()):
    for node_two_id, node_two in dict_nodes.items():
        Id_y.append(node_one_id)
        Lon_y.append(node_one.coordinate[0])
        Lat_y.append(node_one.coordinate[1])
        Id_x.append(node_two_id)
        Lon_x.append(node_two.coordinate[0])
        Lat_x.append(node_two.coordinate[1])
        if (node_one_id, node_two_id) in dict_routes:
            continue
        if node_one_id == node_two_id:
            current_distance.append(sys.float_info.max)
            shortest_route.append([])
        else:
            orig_node = ox.nearest_nodes(G, node_one.coordinate[0], node_one.coordinate[1])
            dest_node = ox.nearest_nodes(G, node_two.coordinate[0], node_two.coordinate[1])
            try:
                current_distance.append(0.001*nx.shortest_path_length(G, orig_node, dest_node, method='bellman-ford', weight='length'))
                shortest_route.append(nx.shortest_path(G, orig_node, dest_node, method='bellman-ford'))
            except nx.NetworkXNoPath:
                current_distance.append(sys.float_info.max)
                shortest_route.append([])
route_list = pd.DataFrame(
    {'Id_x': Id_x,
     'Lon_x': Lon_x,
     'Lat_x': Lat_x,
     'Id_y': Id_y,
     'Lon_y': Lon_y,
     'Lat_y': Lat_y,
     'current_distance': current_distance,
     'shortest_route': shortest_route
     })
route_list.to_csv(".\\Data\\new_Paris.routes_1.csv",  index=False)
