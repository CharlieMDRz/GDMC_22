from typing import Callable, List

from sortedcontainers import SortedList

from parameters import MAX_INT
from utils import Point, manhattan, index_argmin
from utils.algorithms.graphs import Graph, Tree, GridGraph
from utils.misc_objects_functions import numba_list, sections


class AStar:
    def __init__(
            self,
            source: Point,
            target: Point,
            graph: Graph = None,
            heuristic: Callable[[Point, Point], float] = None,
            itinerary: List[Point] = None
    ):
        """
        Instantiates A* problem
        :param source: first extremity of the path
        :param target: last extremity of the path
        :param graph: valid transitions and distances
        :param heuristic: heuristic distance between two points, defaults to manhattan
        :param itinerary: optional intermediary points to fasten computation
        """
        self._source: Point = source
        self._target: Point = target
        self._graph = GridGraph(False) if graph is None else graph
        self._heuristic_dist = manhattan if heuristic is None else heuristic
        self._itinerary = [source, target] if itinerary is None else itinerary
        if self._itinerary[0] != source:
            self._itinerary.insert(0, source)
        if self._itinerary[-1] != target:
            self._itinerary.insert(-1, target)

        # computation variables
        self.__path: List[Point] = []
        self.__predecessor: Tree = Tree()
        self.__predecessor.addEdge(self._source, self._source, 0)
        self.__heuristic_list = self.__init_heuristic_list()  # index -> heuristic from itinerary[index] to the target
        self.__heuristic_index_map = {}  # point -> index of the next closest point in the itinerary

    @property
    def path(self):
        if not self.__path:
            self.__path = self.__compute()
        return self.__path

    def __compute(self) -> List[Point]:
        source, target = self._source, self._target
        neighbours: SortedList = SortedList([source], lambda pos: self.__heuristic(pos))
        heuristic_step = 0

        while neighbours:
            node = neighbours.pop(0)

            if node == target:
                return self.__predecessor.getPathTowards(target)
            elif self.__heuristic_index(node) < heuristic_step:
                continue
            else:
                heuristic_step = max(heuristic_step, self.__heuristic_index(node))

            node_dist = self.__predecessor[node]

            for neigh in self._graph.getNeighbours(node, parent=self.__predecessor.getParent(node), target=target):
                edge_dist = node_dist + self._graph[node, neigh]

                if edge_dist >= MAX_INT:
                    continue

                if neigh in self.__predecessor:
                    if edge_dist < self.__predecessor[neigh]:
                        neighbours.discard(neigh)  # update value
                        self.__predecessor.addEdge(node, neigh, edge_dist)
                        neighbours.add(neigh)
                elif self.__heuristic_index(neigh) >= heuristic_step:
                    self.__predecessor.addEdge(node, neigh, edge_dist)
                    neighbours.add(neigh)

        return []

    def __heuristic_index(self, node) -> int:
        if node not in self.__heuristic_index_map:
            distance_to_path = [manhattan(node, itinerary_node) for itinerary_node in self._itinerary]
            index = index_argmin(numba_list(distance_to_path)) + 1
            self.__heuristic_index_map[node] = min(index, len(self._itinerary) - 1)
        return self.__heuristic_index_map[node]

    def __heuristic(self, pos) -> float:
        cost_to_pos = self.__predecessor[pos]
        pos_step = self.__heuristic_index(pos)
        cost_from_pos = self._heuristic_dist(pos, self._itinerary[pos_step]) + self.__heuristic_list[pos_step]
        return cost_to_pos + cost_from_pos

    def __init_heuristic_list(self):
        itinerary = reversed(self._itinerary)
        heuristic_list = [0]  # target -> null heuristic
        for first_point, second_point in sections(itinerary):
            # for each following sections, increments the heuristic of #i with distance from #i to #i+1
            heuristic_list.append(heuristic_list[-1] + self._heuristic_dist(first_point, second_point))
        # finally, heuristic from the start is the sum of distances between consecutive itinerary points
        return list(reversed(heuristic_list))
