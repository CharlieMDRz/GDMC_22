import math
import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from pathfinding.rail_network import RailRoadGraph
from utils import Point, euclidean


def gen_rail_segment(segment_length) -> Tuple[Point, Point]:
    node = Point(60, 60)
    root_angle = random.random() * math.pi * 2
    root = node + Point(math.cos(root_angle), math.sin(root_angle)) * segment_length
    return root, node


def plot_segment_neighbours(root: Point, node: Point, graph: RailRoadGraph):
    neighbours = graph.getNeighbours(node, parent=root)
    points = list(neighbours) + [root, node]

    min_x = min(p.x for p in points) - 2
    max_x = max(p.x for p in points) + 2
    min_y = min(p.z for p in points) - 2
    max_y = max(p.z for p in points) + 2

    fig, ax = plt.subplots()
    ax.scatter(*root.xz, c='r')
    ax.scatter(*node.xz, c='b')
    ax.scatter([_.x for _ in neighbours], [_.z for _ in neighbours])
    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    fig.show()


def rail_curvature_test():
    graph = RailRoadGraph(7, 15)
    for _ in range(10):
        root, node = gen_rail_segment(7)
        plot_segment_neighbours(root, node, graph)
        graph.getNeighbours(node, parent=root)
        plt.close()


def random_hermit_test():
    p0, p1 = [Point(*np.random.randint(0, 10, 2)) for _ in range(2)]
    q0, q1 = [Point(*(np.random.random(2) * euclidean(p0, p1))) for _ in range(2)]
    plot_hermit(p0, q0, p1, q1)


def turnaround_hermit_test():
    p0, p1 = Point(0, 0), Point(10, 0)
    q0, q1 = Point(0, 10), Point(0, 10)
    plot_hermit(p0, q0, p1, q1)


def plot_hermit(*points):
    assert len(points) == 4
    from pathfinding.rail_network import hermit_curve
    hermit = hermit_curve(*points, False)
    fig, ax = plt.subplots()
    ax.scatter([_.x for _ in hermit], [_.z for _ in hermit])
    plt.show()


if __name__ == '__main__':
    # rail_curvature_test()
    # random_hermit_test()
    turnaround_hermit_test()
