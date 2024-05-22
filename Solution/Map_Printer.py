import folium
import osmnx as ox
import networkx as nx

LOCATIONS = {
    "Paris": [48.864716, 2.349014],
    "Shanghai": [31.224361, 121.469170],
    "LA": [33.918699, -118.324843],
    'USA': [33.918699, -118.324843]
}

COLOR_LIST = [
    "#9400D3",  # Violet
    "#FF7F00",  # Orange
    "#4B0082",  # Indigo
    "#023020",  # Green
    "#0000FF",  # Blue
    "#FFFF00",  # Yellow
    "#FF0000",  # Red
    '#b2296d',  # Pink
    "#696969",  # Grey
    "#666A8C",  # Blue-ish gery
    '#00FFFF',  # Aqua
    '#F5F5DC',  # Beige
    '#ED9121',  # Carrot
    '#00CDCD',  # Cyan
    '#8B7500',  # Gold
    '#CDC9A5',  # Lemonchiffon
    '#FAF0E6',  # Linen
    "#9400D3",  # Violet
    "#FF7F00",  # Orange
    "#4B0082",  # Indigo
    "#023020",  # Green
    "#0000FF",  # Blue
    "#FFFF00",  # Yellow
    "#FF0000",  # Red
    '#b2296d',  # Pink
    "#696969",  # Grey
    "#666A8C",  # Blue-ish gery
    '#00FFFF',  # Aqua
    '#F5F5DC',  # Beige
    '#ED9121',  # Carrot
    '#00CDCD',  # Cyan
    '#8B7500',  # Gold
    '#CDC9A5',  # Lemonchiffon
    '#FAF0E6',  # Linen
]

CLUSTER_COLOR_LIST = [
    # 'lightgreen', 
    'green',
    'darkpurple',
    'darkgreen',
    'orange',
    'blue',
    # 'white',
    'purple',
    # 'pink',
    'black',
    'cadetblue',
    # 'lightgray',
    'darkred',
    'darkblue',
    'gray',
    'lightblue',
    'red',
    'beige',
    'lightred'
]

def easy_print_nodes(network, location, file_name):
    # Create map object
    m = folium.Map(location=LOCATIONS[location], zoom_start=12)
    # Create markers
    for id in list(network.nodes.keys()):
        if id.startswith('D'):
            folium.Marker(
                [network.nodes[id].coordinate[1], network.nodes[id].coordinate[0]],
                popup='<strong>' + str(id) + '</strong>',
                icon=folium.Icon(color='black', icon='home')
            ).add_to(m)
            continue

        folium.CircleMarker(
            (network.nodes[id].coordinate[1], network.nodes[id].coordinate[0]),
            popup='<strong>' + str(id) + '</strong>',
            color="#8e2300",
            fill=True,
            fill_color="#3140cc",
            radius=3,
            fill_opacity=1
        ).add_to(m)

    m.save(f'.\\Solution\\Map\\{file_name}.html')

def easy_print_tours(network, tours, file_name):
    # Create map object
    m = folium.Map(location=LOCATIONS["Paris"], zoom_start=12)

    # Create layer
    shapesLayer = folium.FeatureGroup(name="Vector Shapes").add_to(m)
    for vehicle, route in tours.items():
        if vehicle in [0, 3, 6]:
            tour_color = "red"
        elif vehicle in [1, 4, 7]:
            tour_color = "green"
        elif vehicle in [2, 5, 8]:
            tour_color = "indigo"
        else:
            tour_color = "blue"

        for node_from, node_to in route:
            folium.PolyLine(
                [
                    (network.nodes[node_from].coordinate[1],
                     network.nodes[node_from].coordinate[0]),
                    (network.nodes[node_to].coordinate[1],
                     network.nodes[node_to].coordinate[0])
                ],
                color=tour_color,
                weight=3
            ).add_to(shapesLayer)

    # Create markers
    for id in list(network.nodes.keys()):
        if id == 'D0':
            folium.Marker(
                [network.nodes[id].coordinate[1], network.nodes[id].coordinate[0]],
                popup='<strong>' + str(id) + '</strong>',
                icon=folium.Icon(color='black', icon='home')
            ).add_to(m)
            continue

        folium.CircleMarker(
            (network.nodes[id].coordinate[1], network.nodes[id].coordinate[0]),
            popup='<strong>' + str(id) + '</strong>',
            color="#8e2300",
            fill=True,
            fill_color="#3140cc",
            radius=2,
            fill_opacity=1
        ).add_to(m)

    m.save(f'.\\Solution\\Map\\{file_name}.html')


def osm_print_tours(network, G, result_dict, color_list, file_name):
    m = folium.Map(location=LOCATIONS["Paris"], zoom_start=3)

    # add routes
    for vehicle_id, routes in result_dict['solution'].items():
        for (from_node, to_node) in routes:
            node_one = network.nodes[from_node]
            node_two = network.nodes[to_node]
            orig_node = ox.nearest_nodes(G, node_one.coordinate[0], node_one.coordinate[1])
            dest_node = ox.nearest_nodes(G, node_two.coordinate[0], node_two.coordinate[1])
            try:
                current_path = nx.shortest_path(G, orig_node, dest_node, method='bellman-ford')
                ox.folium.plot_route_folium(G, current_path, m, color=color_list[int(vehicle_id)])
            except (ValueError, nx.NetworkXNoPath):
                # if str(e) == 'graph contains no edges':
                points = [[node_one.coordinate[1], node_one.coordinate[0]],
                          [node_two.coordinate[1], node_two.coordinate[0]]]
                folium.PolyLine(points,
                                color=color_list[int(vehicle_id)],
                                weight=3
                                ).add_to(m)
    # add nodes
    for id in list(network.nodes.keys()):
        if id.startswith('D'):
            folium.Marker(
                [network.nodes[id].coordinate[1], network.nodes[id].coordinate[0]],
                popup='<strong>' + str(id) + '</strong>',
                icon=folium.Icon(color='black', icon="house-chimney", prefix='fa')
            ).add_to(m)
            continue
        folium.CircleMarker(
            (network.nodes[id].coordinate[1], network.nodes[id].coordinate[0]),
            popup='<strong>' + str(id) + '</strong>',
            color="#333333",
            fill=True,
            fill_color="#333333",
            radius=2,
            fill_opacity=1
        ).add_to(m)

    m.save(f'.\\Solution\\Map\\{file_name}.html')


def osm_print_cluster_tours(network, G, cluster_result, result_dict, color_list, file_name):
    m = folium.Map(location=LOCATIONS["LA"], zoom_start=3)
    # add nodes
    for id in [_ for _ in network.nodes.keys() if _.startswith('D')]:
        folium.Marker(
            [network.nodes[id].coordinate[1], network.nodes[id].coordinate[0]],
            popup='<strong>' + str(id) + '</strong>',
            icon=folium.Icon(color='black', icon="house-chimney", prefix='fa')
        ).add_to(m)
    #  Id check
    for p_id, c_id in result_dict['center'].items():
        folium.Marker(
            [network.nodes[c_id].coordinate[1], network.nodes[c_id].coordinate[0]],
            popup='<strong>' + str(p_id) + '</strong>',
            icon=folium.Icon(color=CLUSTER_COLOR_LIST[int(p_id.strip('P'))], icon="warehouse", prefix='fa')
        ).add_to(m)

    for index, (_, station_children) in enumerate(cluster_result.items()):
        for child_id in station_children:
            child_node = network.nodes.get(child_id)
            folium.CircleMarker(
                (child_node.coordinate[1], child_node.coordinate[0]),
                popup='<strong>' + str(child_id) + '</strong>',
                color=CLUSTER_COLOR_LIST[index],
                fill=True,
                fill_color=CLUSTER_COLOR_LIST[index],
                radius=3,
                fill_opacity=1
            ).add_to(m)

    # add routes
    for vehicle_id, routes in result_dict['solution'].items():
        for (from_node, to_node) in routes:
            if from_node.startswith('P'):
                from_node = result_dict['center'].get(from_node)
            if to_node.startswith('P'):
                to_node = result_dict['center'].get(to_node)
            node_one = network.nodes[from_node]
            node_two = network.nodes[to_node]
            orig_node = ox.nearest_nodes(G, node_one.coordinate[0], node_one.coordinate[1])
            dest_node = ox.nearest_nodes(G, node_two.coordinate[0], node_two.coordinate[1])
            try:
                current_path = nx.shortest_path(G, orig_node, dest_node, method='dijkstra')
                ox.folium.plot_route_folium(G, current_path, m, color=color_list[int(vehicle_id)])
            except (ValueError, nx.NetworkXNoPath):
                # if str(e) == 'graph contains no edges':
                points = [[node_one.coordinate[1], node_one.coordinate[0]],
                          [node_two.coordinate[1], node_two.coordinate[0]]]
                folium.PolyLine(points,
                                color=color_list[int(vehicle_id)],
                                weight=3
                                ).add_to(m)

    #             routes.remove([from_node, to_node])
    #             continue
    m.save(f'.\\Solution\\Map\\{file_name}.html')
