# coding=utf-8
import datetime
import json
from pprint import pprint as pretty_print
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


@dataclass
class Dimensions:
    depth: float
    height: float
    width: float

    @property
    def volume(self) -> float:
        return self.depth * self.height * self.width

@dataclass
class TimeWindow:
    start_time: datetime.datetime
    end_time: datetime.datetime

@dataclass
class Package:
    name: str
    time_window: Optional[TimeWindow]
    service_time: float
    dimensions: Dimensions

@dataclass
class Stop:
    name: str
    requested_parcel: List[Package]
    lat: float
    lng: float
    is_depot: bool

@dataclass
class Route:
    name: str
    stops: List[Stop]
    capacity: float
    departure_time: datetime.datetime
    travel_times: Dict[Tuple[str, str], float]

def _parse_dimensions(dimension_data: Dict) -> Dimensions:
    depth_centimeters = dimension_data['depth_cm']
    height = dimension_data['height_cm']
    width = dimension_data['width_cm']
    return Dimensions(depth=depth_centimeters, height=height, width=width)

def _parse_time_window(time_window_data: Dict) -> Optional[TimeWindow]:
    start_time = time_window_data["start_time_utc"]
    end_time = time_window_data["end_time_utc"]
    if isinstance(start_time, float) or isinstance(end_time, float):
        return None

    start_time_date = datetime.datetime.fromisoformat(start_time)
    end_time_date = datetime.datetime.fromisoformat(end_time)

    return TimeWindow(start_time=start_time_date, end_time=end_time_date)

def _parse_package(package_id: str, package_data: Dict) -> Package:
    name = package_id
    time_window = _parse_time_window(package_data['time_window'])
    service_time = package_data['planned_service_time_seconds']
    dimensions = _parse_dimensions(package_data['dimensions'])
    return Package(name=name, time_window=time_window, service_time=service_time, dimensions=dimensions)

def _parse_route_stop(stop_id: str, stop_data: Dict) -> Stop:
    packages = []
    for package_id in stop_data.keys():
        package_data = stop_data[package_id]
        package = _parse_package(package_id, package_data)
        packages.append(package)
    return Stop(name=stop_id, requested_parcel=packages, lat=0, lng=0, is_depot=False)

def _parse_route(route_id: str, route_data: Dict) -> Route:
    stops = []
    for stop_id in route_data.keys():
        stop_data = route_data[stop_id]
        stop = _parse_route_stop(stop_id, stop_data)
        stops.append(stop)
    return Route(name=route_id, stops=stops, capacity=0.0, departure_time=datetime.datetime.now(), travel_times={})

def parse_package_data(eval_package_data_path: str) -> List[Route]:
    with open(eval_package_data_path) as package_data_file:
        package_file_data = json.load(package_data_file)

    routes = []
    for route_id in package_file_data.keys():
        route_data = package_file_data[route_id]
        route = _parse_route(route_id, route_data)
        routes.append(route)

    return routes

def _add_additional_stop_data(stop: Stop, additional_data: Dict):
    lat = additional_data['lat']
    lng = additional_data['lng']
    stop.lat = lat
    stop.lng = lng
    if additional_data['type'] == 'Dropoff':
        stop.is_depot = False
    else:
        stop.is_depot = True

def _add_additional_route_data(route: Route, additonal_data: Dict):
    capacity = additonal_data['executor_capacity_cm3']
    route.capacity = capacity

    departure_time = datetime.datetime.fromisoformat(additonal_data['date_YYYY_MM_DD'] + ' ' + additonal_data['departure_time_utc'])
    route.departure_time = departure_time

    for stop in route.stops:
        stop_id = stop.name
        stop_data = additonal_data['stops'][stop_id]
        _add_additional_stop_data(stop, additional_data=stop_data)

def add_route_data(routes: List[Route], route_data_path: str):
    with open(route_data_path) as route_data_file:
        route_file_data = json.load(route_data_file)

    route_by_id = {}
    for route in routes:
        route_by_id[route.name] = route

    for route_id in route_file_data.keys():
        route_data = route_file_data[route_id]

        route = route_by_id[route_id]

        _add_additional_route_data(route, additonal_data=route_data)

# (AH, AH): distance_between_AH_and_AH
# AH : { AH: distance_between_AH_and_AH, ... }

def _add_travel_time_to_route(route: Route, travel_time_data: Dict):
    travel_times = {}
    for origin_stop in route.stops:
        for target_stop in route.stops:
            travel_times[origin_stop.name, target_stop.name] = travel_time_data[origin_stop.name][target_stop.name]

    route.travel_times = travel_times

def add_travel_times(routes: List[Route], travel_time_data_path: str):
    with open(travel_time_data_path) as travel_time_data_file:
        travel_time_data = json.load(travel_time_data_file)

    for route in routes:
        route_data = travel_time_data[route.name]
        _add_travel_time_to_route(route, route_data)

routes = parse_package_data("./almrrc2021/almrrc2021-data-evaluation/model_apply_inputs/eval_package_data.json")

add_route_data(routes, './almrrc2021/almrrc2021-data-evaluation/model_apply_inputs/eval_route_data.json')
add_travel_times(routes, './almrrc2021/almrrc2021-data-evaluation/model_apply_inputs/eval_travel_times.json')
