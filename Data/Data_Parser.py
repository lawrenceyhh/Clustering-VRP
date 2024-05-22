import pandas as pd
import sys
from geopy import distance
import osmnx as ox
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from tqdm.notebook import tqdm


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
    shortest_path: Optional[List[int]]


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
    variable_cost: float


# def parse_shanghai_data(network_name: str, data_path: str) -> Network:
#     """Parser for the Shanghai dataset

#     Args:
#         network_name (str): name of the network
#         data_path (str): path to [Data] Parcel Deliveries - Shanghai.csv file

#     Returns:
#         Network: a Network object with one nodes dictionary and one route dictionary
#     """
#     data = pd.read_csv(data_path)
#     dict_nodes = {}
#     for i in tqdm(data.index):
#         current_node = Node(
#             data.loc[i, "CUSTOMER"], (data.loc[i, "Y"], data.loc[i, "X"]), data.loc[i, "DEMAND"])
#         if current_node.id not in dict_nodes.keys():
#             dict_nodes[current_node.id] = current_node
#     dict_routes = {}
#     for node_one_id, node_one in tqdm(dict_nodes.items()):
#         for node_two_id, node_two in dict_nodes.items():
#             if (node_one_id, node_two_id) in dict_routes:
#                 continue
#             if node_one_id == node_two_id:
#                 current_distance = sys.float_info.max
#             else:
#                 current_distance = distance.distance(
#                     node_one.coordinate, node_two.coordinate).km
#             dict_routes[(node_one_id, node_two_id)] = Route(
#                 node_one_id, node_two_id, current_distance)
#     return Network(network_name, dict_nodes, dict_routes)


def parse_paris_data(network_name: str, node_data_path: str, route_data_path: str) -> Network:
    """Parser for the Paris dataset

    Args:
        network_name (str): name of the network
        node_data_path (str): path to Paris.nodes
        route_data_path (str): path to Paris.routes

    Returns:
        Network: a Network object with one nodes dictionary and one route dictionary
    """
    node_data = pd.read_csv(node_data_path, sep=" ")
    route_data = pd.read_csv(route_data_path, sep="\t")
    dict_nodes = {}
    for i in tqdm(node_data.index):
        current_node = Node(node_data.loc[i, "Id"], (node_data.loc[i, "Lon"], node_data.loc[i, "Lat"]),
                            node_data.loc[i, "Demand[kg]"])
        if current_node.id not in dict_nodes.keys():
            dict_nodes[current_node.id] = current_node
    dict_routes = {}
    for j in tqdm(route_data.index):
        node_one_id = route_data.loc[j, "Id_x"]
        node_two_id = route_data.loc[j, "Id_y"]
        if "shortest_route" in route_data.columns:
            current_route = Route(node_one_id, node_two_id, route_data.loc[j, "distance"],
                                  route_data.loc[j, "shortest_route"])
        else:
            current_route = Route(node_one_id, node_two_id,
                                  route_data.loc[j, "distance"], None)
        if (node_one_id, node_two_id) not in dict_routes.keys():
            dict_routes[(node_one_id, node_two_id)] = current_route
    return Network(network_name, dict_nodes, dict_routes)


def parse_LA_data(network_name: str, node_data_path: str, route_data_path: str) -> Network:
    """Parser for the LA dataset

    Args:
        network_name (str): name of the network
        node_data_path (str): path to Paris.nodes
        route_data_path (str): path to Paris.routes

    Returns:
        Network: a Network object with one nodes dictionary and one route dictionary
    """
    node_data = pd.read_csv(node_data_path, sep="\t")
    route_data = pd.read_csv(route_data_path, sep="\t")
    dict_nodes = {}
    for i in tqdm(node_data.index):
        current_node = Node(node_data.loc[i, "Id"], (node_data.loc[i, "Lon"], node_data.loc[i, "Lat"]),
                            node_data.loc[i, "Demand"] / 5000)
        if current_node.id not in dict_nodes.keys():
            dict_nodes[current_node.id] = current_node
    dict_routes = {}
    for j in tqdm(route_data.index):
        node_one_id = route_data.loc[j, "Id_x"]
        node_two_id = route_data.loc[j, "Id_y"]
        if "shortest_route" in route_data.columns:
            current_route = Route(node_one_id, node_two_id, route_data.loc[j, "distance"],
                                  route_data.loc[j, "shortest_route"])
        else:
            current_route = Route(node_one_id, node_two_id,
                                  route_data.loc[j, "distance"], None)
        if (node_one_id, node_two_id) not in dict_routes.keys():
            dict_routes[(node_one_id, node_two_id)] = current_route
    return Network(network_name, dict_nodes, dict_routes)


def parse_Shanghai_data(network_name: str, node_data_path: str, route_data_path: str) -> Network:
    """Parser for the Shanghai dataset

    Args:
        network_name (str): name of the network
        node_data_path (str): path to Paris.nodes
        route_data_path (str): path to Paris.routes

    Returns:
        Network: a Network object with one nodes dictionary and one route dictionary
    """
    node_data = pd.read_csv(node_data_path, sep="\t")
    route_data = pd.read_csv(route_data_path, sep="\t")
    dict_nodes = {}
    for i in tqdm(node_data.index):
        current_node = Node(node_data.loc[i, "Id"], (node_data.loc[i, "Lon"], node_data.loc[i, "Lat"]),
                            node_data.loc[i, "Demand"] * 10)
        if current_node.id not in dict_nodes.keys():
            dict_nodes[current_node.id] = current_node
    dict_routes = {}
    for j in tqdm(route_data.index):
        node_one_id = route_data.loc[j, "Id_x"]
        node_two_id = route_data.loc[j, "Id_y"]
        if "shortest_route" in route_data.columns:
            current_route = Route(node_one_id, node_two_id, route_data.loc[j, "distance"],
                                  route_data.loc[j, "shortest_route"])
        else:
            current_route = Route(node_one_id, node_two_id,
                                  route_data.loc[j, "distance"], None)
        if (node_one_id, node_two_id) not in dict_routes.keys():
            dict_routes[(node_one_id, node_two_id)] = current_route
    return Network(network_name, dict_nodes, dict_routes)


def get_common_truck():
    # data referred from https://www.tdbg.de/en/cu-service/delivery-vehicles-and-containers/
    # and https://www.eurodriveuk.com/vehicles/truck-hire/
    big_truck = Truck("Lorry 7.5 t", 2400, 180.00, 1.831/1.609)
    medium_truck = Truck("Delivery vehicles", 1200, 135.00, 1.856/1.609)
    small_truck = Truck("Box van", 400, 65.00, 1.880/1.609)
    return {big_truck.model: big_truck, medium_truck.model: medium_truck, small_truck.model: small_truck}


def fetch_G(location):
    _G = ox.graph.graph_from_xml(filepath=f'.\\Data\\{location}.osm')
    return _G


def fetch_all_dataset(location):
    if location == "Paris":
        parse_data = parse_paris_data
    elif location == "LA":
        parse_data = parse_LA_data
    elif location == "Shanghai":
        parse_data = parse_Shanghai_data

    _data = parse_data(
        f"{location}_network", f".\\Data\\{location}.nodes", f".\\Data\\{location}.routes")
    # _G = ox.graph.graph_from_xml(filepath=f'.\\Data\\{location}.osm')

    _vehicles = get_common_truck()
    _nodes = list(_data.nodes.keys())

    _vertices = _nodes

    # demand
    _demands = {i: _data.nodes[i].demand for i in _nodes}

    # define the distance matrix as arcs
    _arcs = {
        (i, j): _data.routes[i, j].total_distance for i in _vertices for j in _vertices if i != j}

    # define the depot node V_d
    _V_d = [v for v in _vertices if v.startswith("D")]
    # define other customer nodes V_c
    _V_c = [v for v in _vertices if v not in _V_d]

    # assume there are only three types of vehicles
    _K = list(_vehicles.keys())

    # define the capacity of all three types of vehicles
    _Q = [_vehicles[k].capacity for k in _K]

    # assume fixed costs are the same for all types of vehicles
    _F = [_vehicles[k].fix_cost for k in _K]

    # define variable costs for different types of vehicles
    _alpha = [_vehicles[k].variable_cost for k in _K]

    _K = list(range(3))

    return _data, _vertices, _arcs, _V_d, _V_c, _F, _alpha, _K, _Q, _demands
