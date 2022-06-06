import logging
import time
from math import ceil
from typing import List

from numpy import full

from utils import BuildArea, argmin, euclidean, Position, manhattan, log_exec_time
from utils.algorithms.graphs import Graph, Tree, dijkstra
from . import road_network, AStar


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
        t0 = time.time()
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
        log_exec_time(t0, f"Computed rough path from {source} to {target}")
        return rough_path

    def getPath(self, source: Position, target: Position):
        """
        Gets a suboptimal path from source to target
        :param source: source point
        :param target: target point
        :return:
        """
        t0 = time.time()
        if source == target:
            return [source]
        rough_path = self.getRoughPath(target, source)
        path = AStar(source, target, self.__path_cost_graph, manhattan, rough_path).path
        log_exec_time(t0, f"Computing path from {source} to {target}")
        return path

    def getPathTowards(self, target: Position) -> List[Position]:
        """
        Gets a suboptimal path towards unconnected point. Finds the most suitable road point to connect from
        :param target: point to connect
        :return: path from existing road point toward target
        """
        roughPathTowardsTarget = self.getRoughPath(target)
        roughSource = roughPathTowardsTarget[0]
        _, source = dijkstra(self.__path_cost_graph, roughSource, end_condition=(lambda _: road_network.RoadNetwork().is_road(_)))
        return AStar(source, target, self.__path_cost_graph, manhattan, roughPathTowardsTarget).path

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

    @property
    def gwidth(self):
        return int(ceil(self.__area.width / self.__granularity))

    @property
    def glength(self):
        return int(ceil(self.__area.length / self.__granularity))
