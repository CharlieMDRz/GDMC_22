import itertools
from collections import OrderedDict
from typing import Tuple, List, Set, Callable, Dict

import numba
import numpy as np
from gdpc import worldLoader
from numba import prange

from generation.structure import AREA_STRUCTURE
from terrain import HeightMap
from utils import *
from utils.misc_objects_functions import in_limits


class TreesMap(PointArray):
    __trees: List[List[Tuple[int, int, int]]]
    __tree_distance: np.ndarray = None

    def __new__(cls, level: worldLoader.WorldSlice, height: HeightMap):
        values, trees = detect_trees(level, height)
        obj = super().__new__(cls, values)
        obj.__trees = trees
        obj.__origin = Point(level.rect[0], level.rect[1])

        return obj

    def remove_tree_at(self, position: Point):
        while self[position.xz]:
            tree_index = int(self[position.xz].pop())
            tree = self.__trees[tree_index]
            while tree:
                x, y, z = tree.pop()
                tree_point = Point(x, z, y) + self.__origin
                AREA_STRUCTURE.set(tree_point, BlockAPI.blocks.Air)

    @property
    def tree_distance(self) -> np.ndarray:
        if self.__tree_distance is None:
            tree_distances = []
            for tree in self.__trees:
                if tree:
                    x, _, z = list(tree)[0]  # base trunk block position
                    x_dist = X_ARRAY - x
                    z_dist = Z_ARRAY - z
                    tree_distances.append(abs(x_dist) + abs(z_dist))  # manhattan dist to the tree
            if tree_distances:
                self.__tree_distance = np.minimum.reduce(tree_distances)
            else:
                self.__tree_distance = np.full((self.width, self.length), 1000)

        return self.__tree_distance


def _detect_trunks(level: worldLoader.WorldSlice, height: HeightMap) -> List[Set[Tuple[int, int, int]]]:
    # detect trunks
    trees: List[Set[Tuple[int, int, int]]] = []
    trunk_2D_coords: Set[Tuple[int, int]] = set()

    for xz in prange(height.width * height.length):
        x = xz // height.length
        z = xz % height.length
        y = height[x, z] + 1
        block = getBlockRelativeAt(level, x, y, z)
        block_neighbours = itertools.product(range(-1, 2), range(-1, 2))
        if _is_trunk(block) and all((x + dx, z + dz) not in trunk_2D_coords for dx, dz in block_neighbours):
            trees.append({(x, y, z)})
            trunk_2D_coords.add((x, z))

    return trees


def detect_trees(level: worldLoader.WorldSlice, height: HeightMap):
    width, length = height.width, height.length
    values = np.full((width, length), None)

    trees = _detect_trunks(level, height)

    explore(level, trees, trunk_neighbours, _is_trunk)
    explore(level, trees, leaf_neighbours, is_leaf)

    for tree_id, tree_blocks in enumerate(trees):
        for x, _, z in tree_blocks:
            if values[x, z] is None:
                values[x, z] = set()
            values[x, z].add(tree_id)

    return values, trees


def explore(
        level: worldLoader.WorldSlice,
        structure: List[Set[Tuple[int, int, int]]],
        get_neighbours: Callable,
        is_structure_element: Callable
) -> None:
    positions_to_explore: Dict[Tuple[int, int, int], int] = OrderedDict()
    explored_positions: Set[Tuple[int, int, int]] = set()
    tree_id: int
    group: Set[Tuple[int, int, int]]
    for tree_id, group in enumerate(structure):
        positions_to_explore.update({xyz: tree_id for xyz in group})
        explored_positions.update(group)

    while positions_to_explore:
        tree_block = next(iter(positions_to_explore))  # retrieve oldest key from position to explore
        tree_id = positions_to_explore.pop(tree_block)
        for neighbour in get_neighbours(*tree_block).difference(explored_positions):
            explored_positions.add(neighbour)
            # try:
            block_state: str = getBlockRelativeAt(level, *neighbour)
            if is_structure_element(block_state):
                structure[tree_id].add(neighbour)
                positions_to_explore[neighbour] = tree_id
            # except IndexError:
            #     continue


@numba.njit(cache=True)
def _is_trunk(block: str) -> bool:
    return 'log' in block or 'stem' in block


@numba.njit(cache=True)
def is_leaf(block_state: str) -> bool:
    return '_leaves' in block_state or 'mushroom_block' in block_state


def trunk_neighbours(x, y, z):
    wd = BuildArea().width
    ln = BuildArea().length
    return set(_ for _ in itertools.product(range(x-1, x+2), range(y, y+2), range(z-1, z+2)) if in_limits(_, wd, ln))


def leaf_neighbours(x, y, z):
    wd = BuildArea().width
    ln = BuildArea().length
    return set(_ for _ in itertools.product(range(x-1, x+2), range(y-1, y+2), range(z-1, z+2)) if in_limits(_, wd, ln))
