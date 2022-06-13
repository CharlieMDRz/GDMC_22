import logging
import math
import traceback
from typing import List, Set, Dict, Tuple

import numpy as np
from gdpc import lookup

from parameters import RAIL_ROAD_SPACING, RAIL_ROAD_PENALTY
from utils import Position, Point, manhattan, euclidean, Direction, argmin, Singleton, intersect, manhattan2d, pos_bound
from utils.algorithms.graphs import GridGraph
from .path_finder import PathFinder
from .railroad_generator import RailRoadGenerator
from .railway import RailConnector, TrainStation, hermit_curve, RailWay
from .road_network import RoadNetwork, road_build_cost, MAX_INT, high_penalty_on_slopes

__all__ = [
    'RailNetwork',
    'RailRoadGraph'
]

all_but_rails = [_ for _ in lookup.BLOCKS if ("Rail" not in str(_) and "Redstone" not in str(_))]


class RailNetwork(RoadNetwork, metaclass=Singleton):
    __railPathFinder: PathFinder

    def __init__(self, width, length, mc_map=None):
        super().__init__(width, length, mc_map)
        self.__maps = mc_map
        self.__limits = (width, length)
        self.__stations: Dict[Tuple[int, int], TrainStation] = {}
        self.__railPathFinder = PathFinder(7, RailRoadGraph(7, 15, cost=rail_road_build_cost), GridGraph(True, step=7, cost=high_penalty_on_slopes))
        self.track_obstacle = np.zeros((width, length))

    def add_station(self, position: Position, orientation: Direction):
        train_station: TrainStation = TrainStation(position, orientation)
        self.__stations[position.xz] = train_station
        p, q = train_station.connectors[0].pos, train_station.connectors[1].pos
        path = [(p * (1-t) + q * t).asPosition for t in np.linspace(0, 1, int((p-q).norm)+1)]
        super().create_road(p, q, path=path)
        return train_station

    def add_edge(self, p1: Position, p2: Position):
        station1 = self.__stations[p1.xz]
        station2 = self.__stations[p2.xz]
        connector_1_2: RailConnector = station1.connect(station2.position)
        connector_2_1: RailConnector = station2.connect(station1.position)
        if connector_1_2 is not None and connector_2_1 is not None:
            connector_1_2.branch(RailWay(connector_1_2, connector_2_1))
            connector_2_1.branch(RailWay(connector_1_2, connector_2_1))
            # self.create_road(connector_1_2.pos, connector_2_1.pos)

    def create_roads(self):
        # untangle all stations
        for station in self.__stations.values():
            connector1, connector2 = station.connectors  # type: RailConnector, RailConnector
            prev_station: RailConnector = connector1.other_neighbour(connector2)
            next_station: RailConnector = connector2.other_neighbour(connector1)
            if prev_station and next_station and intersect((connector1.position, prev_station.position),
                                                           (connector2.position, next_station.position)):
                connector1.adjacent_connectors = {connector2, next_station}
                connector2.adjacent_connectors = {connector1, prev_station}

        # compute all paths joining connectors
        built_connections = set()
        for station in self.__stations.values():
            connector1, connector2 = station.connectors  # type: RailConnector, RailConnector
            prev_station: RailConnector = connector1.other_neighbour(connector2)
            next_station: RailConnector = connector2.other_neighbour(connector1)
            if prev_station is not None and prev_station not in built_connections:
                self.create_road(prev_station.position, connector1.position)
                built_connections.update({prev_station, connector1})
            if next_station is not None and next_station not in built_connections:
                self.create_road(next_station.position, connector2.position)
                built_connections.update({next_station, connector2})

    def get_rail_direction(self, connector_pos: Position):
        station: TrainStation = argmin(self.__stations.values(), lambda stat: euclidean(connector_pos, stat.position))
        vec: Point = connector_pos - station.position
        vec *= abs(station.orientation.value)  # keeps component along the station axis
        connector_dir = Direction.of(*vec.xyz)
        return connector_dir

    def get_road_width(self, x: Point or int, z: int = None) -> int:
        return 5

    def create_road(self, root_point=None, ending_point=None, path=None):
        if path is None:
            logging.info(f"Creating rail way between {root_point} and {ending_point}")
            rail_path: List[Position] = self.__railPathFinder.getPath(root_point, ending_point)
            rail_path = [Position(p.x, p.z, self.__maps.height_map[p.x, p.z]) for p in rail_path]
            path = [root_point]

            adjustment_index = 4
            adjust_path_at_ending_point = len(rail_path) > adjustment_index
            for i in range((len(rail_path)-adjustment_index) if adjust_path_at_ending_point else (len(rail_path)-1)):
                # section nodes
                cur_start = rail_path[i]
                cur_exit = rail_path[i+1]
                past_start = rail_path[i-1] if i > 0 else cur_start
                next_exit = rail_path[i+2] if (i+2) < len(rail_path) else cur_exit

                # section direction
                start_dir = (cur_exit - past_start) / 3
                end_dir = (next_exit - cur_start) / 3

                path.extend(hermit_curve(cur_start, start_dir, cur_exit, end_dir)[1:])
            from matplotlib import pyplot as plt
            plt.scatter(*np.where(self.network > 0))
            plt.plot([p.x for p in path], [p.z for p in path])

            if adjust_path_at_ending_point:
                adjust_point = rail_path[-adjustment_index]
                adjust_length = manhattan2d(adjust_point, ending_point)

                ending_dir: Point = -self.get_rail_direction(ending_point).value
                ending_dir = ending_dir / ending_dir.norm * adjust_length

                adjust_dir = adjust_point - rail_path[-adjustment_index - 1]
                adjust_dir = adjust_dir / adjust_dir.norm * adjust_length / 3
                adjusted_curve = hermit_curve(adjust_point, adjust_dir, ending_point, ending_dir)[1:]
                plt.plot([p.x for p in adjusted_curve], [p.z for p in adjusted_curve])
                path.extend(adjusted_curve)

            plt.savefig('_'.join(map(str, [root_point.x, root_point.z, ending_point.x, ending_point.z])) + '.png', dpi=300)

            # if manhattan(root_point, ending_point) > TrainStation.PLATFORM_LENGTH:
            if True:
                for track_pos in path:
                    x0, x1 = pos_bound(track_pos.x - 3), pos_bound(track_pos.x + 4, self.length)
                    z0, z1 = pos_bound(track_pos.z - 3), pos_bound(track_pos.z + 4, self.length)
                    self.track_obstacle[x0:x1, z0:z1] = 1

        return super().create_road(path=path)

    def connect_to_network(self, target: Position, margin: int = 0) -> List[Set[Point]]:
        raise NotImplementedError()

    def try_to_create_direct_path(self, node1: Point, node2: Point) -> (List[Point], List[Point]):
        return [], []

    def generate(self, level, districts):
        try:
            for train_station in self.__stations.values():  # type: TrainStation
                train_station.generate(level, self.__maps.height_map)
            RailRoadGenerator(self.network > 0, self.__stations.values()).generate(level, self.__maps.height_map)
        except Exception:
            traceback.print_exc()


def compute_train_line(network: RailNetwork, stations: Set[Position]):
    # Compute distance array
    stations_list: List[Position] = list(stations)
    n_stations = len(stations_list)
    distance_array = np.zeros((n_stations, n_stations))
    for i, station_i in enumerate(stations_list[:-1]):
        for j, stations_j in enumerate(stations_list):
            if j > i:
                distance_array[i, j] = distance_array[j, i] = manhattan(station_i, stations_j)

    stations_to_connect = set(range(1, n_stations))
    connected_stations = {0}

    while stations_to_connect:
        # compute distances
        neighbour = {i: argmin(connected_stations, key=(lambda j: distance_array[i, j])) for i in stations_to_connect}
        distances = {i: distance_array[i, neighbour[i]] for i in neighbour}
        # find best candidate
        i = argmin(distances.keys(), distances.__getitem__)
        j = neighbour[i]
        # update values
        stations_to_connect.remove(i)
        connected_stations.add(i)
        if distance_array[i, j] > 32:
            network.create_road(stations_list[i], stations_list[j])
        else:
            stations_list[i] = stations_list[j]


class RailRoadGraph(GridGraph):
    def __init__(self, section_length: int, curvature: float, **kwargs):
        super().__init__(True, **kwargs)
        self.__length = section_length
        self.__curvature = curvature
        self.__max_dev = math.pi - 2 * math.acos(section_length / (2 * curvature))

    def getNeighbours(self, node, **kwargs):
        def prev_angle(dx, dz):
            if dx == 0 and dz == 0:
                start_dir = RailNetwork().get_rail_direction(node)
                return {
                    Direction.East: 0,
                    Direction.South: math.pi / 2,
                    Direction.West: math.pi,
                    Direction.North: -math.pi / 2
                }[start_dir]

            return math.atan2(dz, dx)
        root = kwargs.get("parent")
        prev_section: Point = node - root
        prev_direction = prev_angle(*prev_section.xz)
        min_direction = prev_direction - self.__max_dev
        max_direction = prev_direction + self.__max_dev

        if kwargs.get('target', False):
            target = kwargs.get('target')
            if euclidean(node, target) <= self.__length:
                return {target}

        neighbours = set()
        for angle in np.linspace(min_direction, max_direction, int(3 * self.__length * self.__max_dev)):
            arc_point: Point = node + Point(math.cos(angle), math.sin(angle)) * self.__length
            try:
                neighbours.add(Position(arc_point.x, arc_point.z))
            except ValueError:
                print(node, prev_section, prev_direction, angle, arc_point)

        neighbours = {_ for _ in neighbours if (2 <= _.x < self.width-2 and 2 <= _.z < self.length-2)}
        return neighbours


def rail_road_build_cost(src_point: Position, dst_point: Position):
    scale = manhattan(src_point, dst_point)
    cost = road_build_cost(src_point, dst_point)

    rail_net: RailNetwork = RailNetwork()
    if rail_net.track_obstacle[dst_point.xz] > 0:
        return MAX_INT

    if cost <= MAX_INT:
        road_net: RoadNetwork = RoadNetwork()
        road_dist = road_net.get_distance(dst_point)
        if road_dist <= RAIL_ROAD_SPACING:
            cost += scale * RAIL_ROAD_PENALTY

    return cost
