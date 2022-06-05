from typing import Dict, List, Tuple, Set, Callable, Union

import numpy as np
from sortedcontainers import SortedList

from parameters import MAX_INT
from utils import Point, BuildArea, manhattan, Direction, Position

__all__ = [
    'connected_component',
    'connected_components',
    'Graph',
    'GridGraph',
    'point_set_as_array',
    'Tree'
]


class Graph:
    def __init__(self, directed, wtype=float):
        self.__adjacency_lists: Dict[Point, Dict[Point, wtype]] = {}
        self.__directed = directed

    def addEdge(self, node, neighbour, weight=None):
        self.__addNode(node)
        self.__addNode(neighbour)
        self.__adjacency_lists[node][neighbour] = weight
        if not self.__directed:
            self.__adjacency_lists[neighbour][node] = weight

    def removeEdge(self, node, neighbour):
        self.__adjacency_lists[node].pop(neighbour)
        if not self.__directed:
            self.__adjacency_lists[neighbour].pop(node)

    def __addNode(self, node):
        if node not in self.nodes:
            self.__adjacency_lists[node] = {}

    @property
    def nodes(self):
        return self.__adjacency_lists.keys()

    def getNeighbours(self, node, **kwargs):
        return set(self.__adjacency_lists[node].keys())

    def __getitem__(self, item):
        node, neigh = item
        return self.__adjacency_lists[node][neigh]

    def __contains__(self, item):
        return item in self.__adjacency_lists


class Tree(Graph):
    def __init__(self):
        super().__init__(True)
        self.__parent_node: Dict[Point, Point] = {}

    def addEdge(self, parent, child, weight=None):
        if child in self.__parent_node:
            self.removeEdge(self.getParent(child), child)
        super().addEdge(parent, child, weight)
        self.__parent_node[child] = parent

    def getParent(self, node):
        if node in self.__parent_node:
            return self.__parent_node[node]
        return None

    def getPathTowards(self, target: Point) -> List[Point]:
        """
        Gets the path in the tree
        :param target: Node in the tree
        :return: path from tree source to target
        """
        node = target  # starts in the target
        path: List[Point] = []
        while node != self.getParent(node):
            path.insert(0, node)  # adds in first Point, keeps the order
            node = self.getParent(node)  # go up in the tree
        path.insert(0, node)  # finally, add the tree source
        return path

    def __getitem__(self, item: Union[Point, Tuple[Point, Point]]):
        if isinstance(item, Point):
            return self[self.getParent(item), item]
        return super().__getitem__(item)


class GridGraph(Graph):
    def __init__(self, directed, **kwargs):
        """
        :param directed: is the graph directed
        :keyword step: granularity of the graph, def = 1
        :keyword width: width of the graph, def: terrain width
        :keyword length: length of the graph, def: terrain length
        :keyword cost: cost function to apply on edges (pair of nodes), def = road build cost
        """
        super().__init__(directed)
        self.step = kwargs.get("step", 1)
        self.width = kwargs.get("width", BuildArea().width)
        self.length = kwargs.get("length", BuildArea().length)
        self.__cost = kwargs.get("cost", manhattan)

    def getNeighbours(self, node, **kwargs):
        res = set()
        for _dir in Direction.cardinal_directions():
            neigh = node + (_dir * self.step)
            if 0 <= neigh.x < self.width and 0 <= neigh.z < self.length:
                res.add(neigh)

        return res

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            node, neigh = item
            cost = self.__cost(node, neigh)
            self.addEdge(node, neigh, cost)
            return cost


def dijkstra(graph: Graph, source: Point or Set[Point], end_condition=(lambda _: False)) -> Tuple[Tree, Point]:
    """
    Dijkstra algorithm
    :param graph: graph to explore
    :param source: starting point of the exploration
    :param end_condition: ending condition on the explored node
    :return: (tree starting in source, last node explored)
    """

    if isinstance(source, Point):
        source = {source}
    source_tree = Tree()
    explored_nodes: Set[Point] = set()
    for source_node in source:
        source_tree.addEdge(source_node, source_node, 0)

    # Stores points to join, sorted by distance to the joined points
    neighbours: SortedList = SortedList(source, lambda pos: source_tree[pos])

    node: Point = source.pop()
    while neighbours:
        node = neighbours.pop(0)
        if end_condition(node):
            break

        elif node in explored_nodes:
            continue

        explored_nodes.add(node)
        node_distance = source_tree[node]
        for neighbour in filter(lambda n: n not in explored_nodes, graph.getNeighbours(node, parent=source_tree.getParent(node))):
            prev_neighbour_distance = source_tree[neighbour] if neighbour in source_tree else MAX_INT
            node_neighbour_distance = node_distance + graph[node, neighbour]
            if node_neighbour_distance < prev_neighbour_distance:
                source_tree.addEdge(node, neighbour, node_neighbour_distance)
                neighbours.add(neighbour)

    return source_tree, node


def connected_component(
        graph: Graph,
        source: Point,
        are_connected: Callable[[Point, Point], bool],
        max_size: int = -1
) -> Set[Point]:
    component: Set[Point] = set()
    points_to_explore: Set[Point] = {source}

    while points_to_explore and max_size:
        new_comp_point = points_to_explore.pop()

        for neighbour in graph.getNeighbours(new_comp_point):
            if are_connected(new_comp_point, neighbour) and neighbour not in component:
                points_to_explore.add(neighbour)

        component.add(new_comp_point)
        max_size -= 1

    return component


def connected_components(points: Set[Position]) -> List[Set[Position]]:
    points_to_explore = points.copy()
    components = []

    while points_to_explore:
        graph = GridGraph(False)
        components.append(connected_component(graph, points_to_explore.pop(), lambda u, v: v in points_to_explore))
        points_to_explore.difference_update(components[-1])

    return components


def point_set_as_array(points: Set[Point]) -> Tuple[Point, np.ndarray]:
    min_x, max_x = min(_.x for _ in points), max(_.x for _ in points)
    min_z, max_z = min(_.z for _ in points), max(_.z for _ in points)

    origin = Point(min_x, min_z)

    width = max_x - min_x + 1
    length = max_z - min_z + 1
    mask = np.full((width, length), False)
    for p in points:
        mask[p.x - min_x, p.z - min_z] = True

    return origin, mask
