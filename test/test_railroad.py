import math
import time
import random
from typing import Tuple

import matplotlib.pyplot as plt

from terrain.rail_network import RailRoadGraph
from utils import Point


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


if __name__ == '__main__':
    graph = RailRoadGraph(7, 15)
    for _ in range(10):
        root, node = gen_rail_segment(7)
        plot_segment_neighbours(root, node, graph)
        graph.getNeighbours(node, parent=root)
        plt.close()
