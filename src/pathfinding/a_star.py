from typing import Callable, List, Set

from sortedcontainers import SortedList

from parameters import MAX_INT
from utils import Position, Point
from utils.algorithms.graphs import Graph, Tree


def a_star(source: Position, target: Position, cost_graph: Graph, heuristic: Callable[[Position], float]) -> List[Point]:
    pred_tree: Tree = Tree()
    # keeps track of predecessor node, edge is weighted with the distance from source to the node
    pred_tree.addEdge(source, source, 0)
    neighbours: SortedList = SortedList([source], lambda pos: pred_tree[pos] + heuristic(pos))

    node = source
    while node != target and neighbours:
        node = neighbours.pop(0)

        node_dist = pred_tree[node]
        for neigh in cost_graph.getNeighbours(node, parent=pred_tree.getParent(node), target=target):
            edge_dist = node_dist + cost_graph[node, neigh]

            if edge_dist >= MAX_INT:
                continue

            if neigh in pred_tree:
                if edge_dist < pred_tree[neigh]:
                    neighbours.discard(neigh)  # update value
                    pred_tree.addEdge(node, neigh, edge_dist)
                    neighbours.add(neigh)
            else:
                pred_tree.addEdge(node, neigh, edge_dist)
                neighbours.add(neigh)

    if target in pred_tree:
        return pred_tree.getPathTowards(target)
    return []
