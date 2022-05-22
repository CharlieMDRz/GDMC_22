from math import ceil
from typing import List

from numpy import full
from sortedcontainers import SortedList

from parameters import MAX_INT
from . import a_star
from .road_network import RoadNetwork, road_build_cost as cost_function
from utils import BuildArea, argmin, euclidean, Position
from utils.algorithms.graphs import Graph, Tree, dijkstra


class PathFinder:
    """
    Path finder. Combines Dijkstra for optimal medium step path, and A* to follow this rough path from origin to destination
    """
    ASTAR_TIME_LIMIT = 15

    def __init__(self, granularity: int, path_cost_graph: Graph, heuristic_graph: Graph):
        self.__area: BuildArea = BuildArea()
        self.__granularity: int = granularity

        self.__has_road = full((self.gwidth, self.glength), False, dtype=bool)

        self.__path_cost_graph: Graph = path_cost_graph
        self.__heuristic_graph: Graph = heuristic_graph

    def getRoughPath(self, target: Position, source: Position = None):
        """
        Finds rough path (with a step > 1) from source to target. Assumes that source is already connected to the road net
        :param target: point to connect, dijkstra will start in this position, and try to walk up to the source, or the road net
        :param source: optional target point
        :return: path, ie list of points, from source (or possible source) to target
        """
        step = self.__granularity
        rough_target = (target - target % step)
        if source is None:
            # If target is None, Dijkstra explores until finding a road point
            end_condition = (lambda _: self.__hasRoad(_))
            target_tree, rough_source = dijkstra(self.__heuristic_graph, rough_target, end_condition)  # type: Tree, Position
            source = rough_source
        else:
            rough_source = source - source % step
            # Otherwise, explores until joining both positions
            end_condition = (lambda _: euclidean(_, rough_source) < self.__granularity)  # ends in rough source
            target_tree, _ = dijkstra(self.__heuristic_graph, rough_target, end_condition)  # type: Tree, Position  # starts in target

        # In both cases, target_tree starts in rough_target
        rough_source = argmin(target_tree.nodes, lambda n: euclidean(n, rough_source))
        targetSourcePath = target_tree.getPathTowards(rough_source)
        sourceTargetPath = list(reversed(targetSourcePath))  # path from source to target
        if len(sourceTargetPath) < 3:
            rough_path = [source, target]
        else:
            rough_path = [source] + sourceTargetPath[1: -1] + [target]

        # Clean up rough angles
        i = 1
        while i < len(rough_path) - 1:
            if (rough_path[i+1] - rough_path[i]).dot(rough_path[i] - rough_path[i-1]) == 0:
                rough_path.pop(i)
            else:
                i += 1
        return rough_path

    def getPath(self, source: Position, target: Position):
        """
        Gets a suboptimal path from source to target
        :param source: source point
        :param target: target point
        :return:
        """
        if source == target:
            return [source]
        rough_path = self.getRoughPath(target, source)
        return self.__astar(source, target, rough_path)

    def getPathTowards(self, target: Position):
        """
        Gets a suboptimal path towards unconnected point. Finds the most suitable road point to connect from
        :param target: point to connect
        :return: path from existing road point toward target
        """
        roughPathTowardsTarget = self.getRoughPath(target)
        roughSource = roughPathTowardsTarget[0]
        _, source = dijkstra(self.__path_cost_graph, roughSource, end_condition=(lambda _: RoadNetwork().is_road(_)))
        return self.__astar(source, target, roughPathTowardsTarget)

    def registerRoad(self, road: List[Position]):
        for p in road:
            self.__setRoad(p)
        pass  # mark road points & update cost graph

    def __setRoad(self, p: Position):
        q = p // self.__granularity
        self.__has_road[q.x, q.z] = True

    def __hasRoad(self, p: Position):
        q = p // self.__granularity
        return self.__has_road[q.x, q.z]

    def __astar(self, source: Position, target: Position, rough_path):
        """
        Custom A* algorithm - computes path with decreasing steps
        :param source: source point (x, z)
        :param target: target point (x, z)
        """

        def build_cumsum() -> List:
            """
            Computes target heuristic for each point in the rough path
            :return:
            """
            l = [0]
            for i in range(len(rough_path) - 1, 0, -1):
                l.append(l[-1] + cost_function(rough_path[i], rough_path[i - 1]))
            return list(reversed(l))

        def init():
            dims = self.__area.width, self.__area.length
            _distance_map = full(dims, MAX_INT, dtype=float)
            _distance_map[source.x, source.z] = 0
            _predecessor_map = full(dims, None)
            _predecessor_map[source.xz] = source
            _heuristic_map = full(dims, MAX_INT, dtype=float)
            return _distance_map, _predecessor_map, _heuristic_map

        def heuristic(_pos):
            if heuristic_map[_pos.x, _pos.z] == MAX_INT:
                _id = argmin(map(lambda p: euclidean(_pos, p), rough_path))  # index of the closest reference point
                # heuristic = distance towards this point + heuristic starting in this point
                if _id >= len(rough_path) - 2:
                    heuristic_map[_pos.xz] = cost_function(_pos, target)
                else:
                    heuristic_map[_pos.xz] = cost_function(_pos, rough_path[_id + 2]) + cumsum[_id + 2]
            return heuristic_map[_pos.xz]

        cumsum = build_cumsum()
        distance_map, predecessor_map, heuristic_map = init()
        return a_star(source, target, self.__path_cost_graph, heuristic)

    @property
    def gwidth(self):
        return int(ceil(self.__area.width / self.__granularity))

    @property
    def glength(self):
        return int(ceil(self.__area.length / self.__granularity))
