# coding=utf-8
import time
from random import choice
from typing import Callable, List, Set

from sortedcontainers import SortedList

import terrain
from generation.generators import *
from generation.road_generator import RoadGenerator
from parameters import *
from utils import Point, euclidean
from utils.algorithms.graphs import GridGraph
from .obstacle_map import ObstacleMap


class RoadNetwork(metaclass=Singleton):
    """
    Road network, computes and stores roads to reach every location. If used correctly, the road network should be
    continuous, ie you can walk from every road point to every other road points by only walking on paths.
    The road network is built simultaneously with the RoadGenerator, which concretely builds the paths
    """
    INSTANCE = None

    def __init__(self, width, length, mc_map=None):
        # type: (int, int, terrain.TerrainMaps) -> RoadNetwork
        self.width = width
        self.length = length
        self.network = np.zeros((width, length), dtype=int)
        # Representing the distance from the network + the path to the network
        self.cost_map = PointArray(np.full((width, length), MAX_INT))
        self.distance_map = PointArray(np.full((width, length), MAX_INT))
        # paths from existing road points to every reachable destination
        self.path_map = PointArray(np.empty((width, length), dtype=list))
        self.lambda_max = MAX_LAMBDA

        # points passed through create_road or connect_to_network
        self.nodes: Set[Position] = set()
        self.road_blocks: Set[Position] = set()
        self.special_road_blocks: Set[Position] = set()
        self.__generator = RoadGenerator(self, mc_map.box, mc_map) if mc_map else None
        self.terrain = mc_map
        RoadNetwork.INSTANCE = self
        from utils.algorithms.path_finder import PathFinder
        self.__pathFinder: PathFinder = PathFinder(6, GridGraph(True, step=1, cost=road_recording_cost), GridGraph(True, step=6, cost=road_build_cost))

    # region GETTER AND SETTER

    def get_road_width(self, x: Point or int, z: int = None) -> int:
        """
        Get road width at specific point
        """
        if z is None:
            return self.get_road_width(x.x, x.z)
        else:
            return self.network[x, z]

    def __get_closest_node(self, point):

        closest_node = None
        min_distance = MAX_FLOAT
        if not self.nodes:
            closest_node = choice(list(self.road_blocks))
        else:
            for node in self.nodes:
                if euclidean(node, point) < min_distance:
                    closest_node = node
                    min_distance = euclidean(node, point)

        if not self.road_blocks.difference(self.nodes):
            return closest_node

        # try to promote an extremity to node
        alt_edge = [_ for _ in self.road_blocks.difference(self.nodes)]
        alt_dist = [euclidean(_, point) for _ in alt_edge]
        i = int(argmin(alt_dist))
        edge, dist = alt_edge[i], alt_dist[i]
        if dist < min_distance:
            if euclidean(edge, closest_node) >= DIST_BETWEEN_NODES:
                self.nodes.add(edge)
            closest_node = edge
        return closest_node

    def get_distance(self, x: Point or int, z: int = None) -> float:
        if z is None:
            return MAX_INT if not ObstacleMap().is_accessible(x) else self.get_distance(x.x, x.z)
        else:
            if self.distance_map[x][z] == MAX_INT:
                return MAX_INT
            return max(0, self.distance_map[x][z] - self.get_road_width(self.get_closest_road_point(Point(x, z))))

    def __set_road_block(self, xp, z=None):
        # type: (Position or int, None or int) -> None
        if z is None:
            # steep roads are not marked as road points
            maps = self.terrain
            if maps and (maps.height_map.steepness(xp.x, xp.z) >= 0.35 or maps.fluid_map.is_water(xp)
                         or not ObstacleMap().is_accessible(xp)):
                self.special_road_blocks.add(xp)
            else:
                self.road_blocks.add(xp)
            x, z = xp.x, xp.z
            if self.network[x, z] == 0:
                self.network[x, z] = MIN_ROAD_WIDTH
            elif self.network[x, z] < MAX_ROAD_WIDTH:
                self.network[x, z] += 1
        else:
            self.__set_road_block(Position(xp, z))

    def is_road(self, x, z=None):
        # type: (Point or int, None or int) -> bool
        if z is None:
            return self.is_road(x.x, x.z)
        else:
            return self.network[x][z] > 0

    def get_closest_road_point(self, point):
        # type: (Point) -> Point
        if self.is_road(point):
            return point
        if self.is_accessible(point):
            return self.path_map[point][0]
        else:
            return self.__get_closest_node(point)

    def __invalidate(self, point):
        self.distance_map[point] = MAX_INT
        self.cost_map[point] = MAX_INT
        self.path_map[point] = None

    def __set_road(self, path):
        # type: ([Point]) -> None
        force_update = False
        if self.__generator:
            self.__generator.handle_new_road(path)
        for point in path:
            self.__set_road_block(point.asPosition)
        self.__update_distance_map(path, force_update)
        self.__pathFinder.registerRoad(path)

    def is_accessible(self, point: Point) -> bool:
        path = self.path_map[point]
        return type(path) == list and (len(path) > 0 or self.is_road(point))

    # region PUBLIC INTERFACE

    def create_road(self, root_point=None, ending_point=None, path=None):
        # type: (Position, Position, List[Position]) -> List[Position]
        if path is None:
            assert root_point is not None and ending_point is not None
            print(f"[RoadNetwork] Compute road path from {str(root_point + self.terrain.area.origin)} "
                  f"towards {str(ending_point + self.terrain.area.origin)}", end="")
            _t0 = time.time()
            path = self.__pathFinder.getPath(root_point, ending_point)
            self.nodes.update({root_point.asPosition, ending_point.asPosition})
            print(f" in {(time.time() - _t0):0.2f}s")
        self.__set_road(path)
        return path

    def connect_to_network(self, target: Position, margin: int = 0) -> List[Set[Point]]:
        """
        Create roads to connect a point to the network. Creates at least one road, and potential other roads (to create
        cycles)
        Parameters
        ----------
        point_to_connect new destination in the network
        margin how close to get to the new point when reaching it

        Returns
        -------
        Created road cycles (possibly empty)
        """

        # safe check: the point is already connected
        if self.is_road(target):
            return []

        # either the path is precomputed or computed with a*
        path: List[Point]
        if self.is_accessible(target) and all(ObstacleMap().is_accessible(_) for _ in self.path_map[target]):
            path = self.path_map[target]
            print(f"[RoadNetwork] Found existing road towards {str(target)}")
        else:
            _t0 = time.time()
            path = self.__pathFinder.getPathTowards(target)
            print(f"[RoadNetwork] Computed road path towards {str(target)} in {(time.time() - _t0):0.2f}s")

        # if a* fails, return
        if not path:
            return []

        # else, register new road(s)
        if margin > 0:
            truncate_index = next(i for i, p in enumerate(path) if manhattan(p, target) <= margin)
            path = path[:truncate_index]
            if not path:
                return []
            target = path[-1]
        self.__set_road(path)
        self.nodes.add(target.asPosition)

        _t1, cycles = time.time(), []
        for node in sorted(self.nodes, key=lambda n: euclidean(n, target))[1:min(CYCLE_ALTERNATIVES, len(self.nodes))]:
            old_path, new_path = self.cycle_creation_condition(node, target)
            if new_path:
                new_path = self.create_road(path=new_path)
                cycles.append(set(old_path).union(set(new_path)))
                if len(cycles) == 2:
                    # allow max 2 new cycles
                    break
        print(f"[RoadNetwork] Computed {len(cycles)} new road cycles in {(time.time() - _t1):0.2f}s")

        if path and all(euclidean(path[0], node) > DIST_BETWEEN_NODES for node in self.nodes):
            self.nodes.add(path[0].asPosition)

        return cycles

    # endregion

    def generate(self, level, districts):
        hm = level.height_map[:]  # type: terrain.HeightMap
        self.__generator.generate(level, hm[:], districts)

    def __update_distance_map(self, road: List[Point], force_update=False):
        self.dijkstra(road, self.lambda_max, force_update)

    # return the path from the point satisfying the ending_condition to the root_point, excluded
    def dijkstra(self, root_points, max_distance, force_update):
        # type: (List[Point], int, bool) -> None
        """
        Accelerated Dijkstra algorithm to compute distance & shortest paths from root points to all others.
        The cost function is the euclidean distance
        Parameters
        ----------
        root_points null distance points to start the exploration
        max_distance max distance from road where to compute dijkstra paths
        force_update ?

        Returns
        -------
        Nothing. Result is stored in self.distance_map and self.path_map
        """

        def init():
            _distance_map = PointArray(np.full((self.width, self.length), MAX_INT))  # on foot distance walking from road points
            _cost_map = PointArray(np.full((self.width, self.length), MAX_INT))  # cost distance building from road points
            for root_point in root_points:
                _distance_map[root_point] = 0
                _cost_map[root_point] = 0
            _neighbours = SortedList(root_points, lambda p: _cost_map[p])
            _predecessor_map = PointArray(np.empty((self.width, self.length), dtype=object))
            return _cost_map, _distance_map, _neighbours, _predecessor_map

        def update_distance(updated_point, neighbor, _neighbors: SortedList):
            edge_cost = road_build_cost(updated_point, neighbor)
            edge_dist = euclidean(updated_point, neighbor)
            if edge_cost == MAX_INT:
                return

            new_cost = cost_map[updated_point] + edge_cost
            new_dist = distance_map[updated_point] + edge_dist
            previous_cost = cost_map[neighbor]
            if previous_cost >= MAX_INT and new_dist <= max_distance and not self.is_road(neighbor) \
                    and (new_cost < self.cost_map[neighbor] or force_update):
                _neighbors.add(neighbor)
            if previous_cost > new_cost:
                cost_map[neighbor] = new_cost
                distance_map[neighbor] = new_dist
                predecessor_map[neighbor] = updated_point

        def update_distances(updated_point):
            x, z = updated_point.x, updated_point.z
            if x + 1 < self.width:
                update_distance(updated_point, Point(x + 1, z), neighbors)
            if x - 1 >= 0:
                update_distance(updated_point, Point(x - 1, z), neighbors)
            if z + 1 < self.length:
                update_distance(updated_point, Point(x, z + 1), neighbors)
            if z - 1 >= 0:
                update_distance(updated_point, Point(x, z - 1), neighbors)

        def path_to_dest(dest_point):
            path = [dest_point]
            while not self.is_road(path[0]):
                path.insert(0, predecessor_map[path[0]])
            return path

        def update_maps_info_at(point):
            x, z = point.x, point.z
            if self.cost_map[x, z] > cost_map[x, z]:
                self.cost_map[point] = cost_map[point]
                self.distance_map[point] = distance_map[point]
                self.path_map[point] = path_to_dest(point)

        cost_map, distance_map, neighbors, predecessor_map = init()
        while neighbors:
            clst_neighbor = neighbors.pop(0)
            update_maps_info_at(clst_neighbor)
            update_distances(clst_neighbor)

    def a_star(self, root_point, ending_point, cost_function, timer=False):
        # type: (Point, Point, Callable[[Point, Point], int], bool) -> List[Point]
        """
        Parameters
        ----------
        root_point path origin
        ending_point path destination
        cost_function (RoadNetwork, Point, Point) -> int

        Returns
        -------
        best first path from root_point to ending_point if any exists
        """
        from utils.algorithms import a_star
        if root_point == ending_point:
            return [root_point]
        t0 = time.time()
        try:
            tuple_path = a_star((root_point.x, root_point.z), (ending_point.x, ending_point.z), (self.width, self.length),
                                lambda u, v: cost_function(Position(u[0], u[1]), Position(v[0], v[1])))
            path = [Point(u, v) for u, v in tuple_path]
        except SystemError or KeyError or ValueError:
            return []
        if timer:
            t0 = time.time() - t0 + .001
            print(f"Fast a*'ed a {len(path)} blocks road in {int(t0) if t0 > 1 else t0} seconds, "
                  f"avg: {int(len(path) / t0)}mps")
        return path

    def cycle_creation_condition(self, node1: Point, node2: Point) -> (List[Point], List[Point]):
        """
        Evaluates whether it's useful to create a new road between two road points
        :param node1:
        :param node2:
        :return:
        """
        straight_dist = euclidean(node1, node2)
        if not (MIN_DISTANCE_CYCLE <= straight_dist <= MAX_DISTANCE_CYCLE):
            return [], []

        existing_path = self.a_star(node1, node2, road_only_cost)
        current_dist = len(existing_path)
        if current_dist / straight_dist < MIN_CYCLE_GAIN:
            return existing_path, []
        straight_path = self.a_star(node1, node2, road_build_cost)
        straight_dist = len(straight_path)
        if straight_dist and current_dist / straight_dist >= MIN_CYCLE_GAIN:
            return existing_path, straight_path
        return existing_path, []

    @property
    def obstacle(self):
        obs = np.full((self.width, self.length), False)
        for p in self.road_blocks.union(self.special_road_blocks):
            x0, z0 = p.x, p.z
            # build a circular obstacle of designated width around road point
            margin = self.get_road_width(x0, z0) / 2 + .5
            for x1, z1 in itertools.product(sym_range(x0, margin, self.width), sym_range(z0, margin, self.length)):
                obs[x1, z1] = True
        return obs

road_build_cache = {}
def road_build_cost(src_point, dest_point):
    network: RoadNetwork = RoadNetwork.INSTANCE
    cost = scale = manhattan(src_point, dest_point)

    # First, a couple safe checks
    # if we don't have access to terrain info
    if network.terrain is None or network.is_road(dest_point) or not cost:
        return cost

    # if dest_point is an obstacle, return inf
    is_dest_obstacle = not ObstacleMap().is_accessible(dest_point)
    is_dest_obstacle |= network.terrain.fluid_map.is_lava(dest_point, margin=MIN_DIST_TO_LAVA)
    if is_dest_obstacle:
        return MAX_INT

    # Then, terrain specific costs, use cache
    if (src_point, dest_point) not in road_build_cache:
        # specific cost to build on water
        if network.terrain.fluid_map.is_water(dest_point, margin=MIN_DIST_TO_RIVER):
            if network.terrain.fluid_map.is_water(src_point):
                return scale * BRIDGE_COST  # bridge continuation
            return BRIDGE_UNIT_COST + (scale - 1) * BRIDGE_COST  # bridge creation

        # discount to get roads closer to water
        src_water = network.terrain.fluid_map.water_distance(src_point)
        dest_water = network.terrain.fluid_map.water_distance(dest_point)
        if 2.5 * MIN_DIST_TO_RIVER >= dest_water > MIN_DIST_TO_RIVER:
            cost += (dest_water - src_water)

        # additional cost for slopes
        direction: Point = (dest_point - src_point).unit
        hm = network.terrain.height_map
        steepness: Point = (hm.steepness(src_point, norm=False) + hm.steepness(dest_point, norm=False)) / 2
        elevation = abs(steepness.dot(direction))
        cost += (1 + elevation) ** 2 - 1  # quadratic cost over slopes

        # discount to cross rail tracks
        from terrain.rail_network import RailNetwork
        rail_road = RailNetwork()
        rail_dist = rail_road.get_distance(dest_point)
        if rail_dist <= RAIL_ROAD_SPACING:
            cost += scale * RAIL_ROAD_PENALTY

        road_build_cache[(src_point, dest_point)] = value = max(scale, cost)
        return value
    return road_build_cache[(src_point, dest_point)]


def road_only_cost(src_point, dest_point):
    network = RoadNetwork.INSTANCE
    return manhattan(src_point, dest_point) if network.is_road(dest_point) else MAX_INT


def road_recording_cost(src_point, dest_point):
    network = RoadNetwork.INSTANCE
    if network.is_road(dest_point):
        return 1
    if network.path_map[src_point] and dest_point in network.path_map[src_point]:
        # Use Dijkstra optimal path to road network if it exists
        return 1
    return road_build_cost(src_point, dest_point)
