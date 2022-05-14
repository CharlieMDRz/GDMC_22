from typing import Tuple, List

from gdpc import worldLoader
import numba
from numba import prange, i8
from numba.core.types import UniTuple, Set as nbSet
import numpy as np

from terrain import HeightMap
from utils import *
from utils.misc_objects_functions import _in_limits


class TreesMap(PointArray):
    __trees: List[List[Tuple[int, int, int]]]
    __tree_distance: np.ndarray = None

    def __new__(cls, level: worldLoader.WorldSlice, height: HeightMap):
        values, trees = _process(level, height)
        obj = super().__new__(cls, values)
        obj.__trees = trees
        obj.__origin = Point(level.rect[0], level.rect[1])

        return obj

    def remove_tree_at(self, position: Point):
        tree_index = int(self[position.xz])
        for x, y, z in self.__trees[tree_index]:
            tree_point = Point(x, z, y) + self.__origin
            setBlock(tree_point, BlockAPI.blocks.Air)
        self.__trees[tree_index] = []

    @property
    def tree_distance(self) -> np.ndarray:
        if self.__tree_distance is None:
            tree_distances = []
            for tree in self.__trees:
                if tree:
                    x, _, z = tree[0]  # base trunk block position
                    x_dist = X_ARRAY - x
                    z_dist = Z_ARRAY - z
                    tree_distances.append(abs(x_dist) + abs(z_dist))  # manhattan dist to the tree
            if tree_distances:
                self.__tree_distance = np.minimum.reduce(tree_distances)
            else:
                self.__tree_distance = np.full((self.width, self.length), 1000)

        return self.__tree_distance


def _detect_trunks(level: worldLoader.WorldSlice, height: HeightMap):
    # detect trunks
    trunks = set()

    for xz in prange(height.width * height.length):
        x = xz // height.length
        z = xz % height.length
        y = height[x, z] + 1
        block = getBlockRelativeAt(level, x, y, z)
        if _is_trunk(block):
            trunks.add((x, z))

    return trunks


def _process(level: worldLoader.WorldSlice, height: HeightMap):
    width, length = height.width, height.length
    values = np.zeros((width, length))

    trunks = _detect_trunks(level, height)

    # Initialize tree structure & propagation
    tree_blocks: List[UniTuple(i8, 2)] = []
    marked_blocks: nbSet(UniTuple(i8, 2)) = set()

    trees = [[]]
    for tree_index, position in enumerate(trunks):
        tree_index += 1  # Start counter to 1
        trees.append([])  # instantiate new tree
        tree_blocks.append((*position, tree_index))  # register trunk
        marked_blocks.add(position)

    # propagate trees through trunks and leaves (and mushrooms)
    while tree_blocks:
        # Get the oldest element in the blocks to process
        x1, z1, tree_index = tree_blocks.pop(0)  # type: int, int, int

        # add it to the structure of its tree
        for y1 in range(height[x1, z1]+1, height.upper_height(x1, z1)+1):
            trees[tree_index].append((x1, y1, z1))
        values[x1, z1] = tree_index

        # check if neighbours for possible other tree points
        for x2 in prange(x1 - 1, x1 + 2):
            for z2 in prange(z1 - 1, z1 + 2):
                if _in_limits((x2, 0, z2), width, length):
                    y2 = height.upper_height(x2, z2)
                    position = (x2, y2, z2)
                    possible_tree_point = (x2, z2)
                    if (possible_tree_point not in marked_blocks) and _is_tree(getBlockRelativeAt(level, *position)):
                        marked_blocks.add(possible_tree_point)
                        tree_blocks.append((*possible_tree_point, tree_index))

    return values, trees


@numba.njit(cache=True)
def _is_trunk(block: str) -> bool:
    return 'log' in block or 'stem' in block


@numba.njit(cache=True)
def _is_tree(bid: str) -> bool:
    return _is_trunk(bid) or '_leaves' in bid or 'mushroom_block' in bid


# @numba.njit(b1(UniTuple(i8, 3), string, nbSet(UniTuple(i8, 3))))
# def _neighbours_not_trees(p0: UniTuple(i8, 3), block0: str, trunks0: Set[UniTuple(i8, 3)]):
#     if block0.startswith("oak") or block0.startswith("birch") or block0.startswith("acacia"):
#         return True
#     x, z, y = p0
#     for dx, dz, dy in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]:
#         if (x + dx, z + dz, y + dy) in trunks0:
#             return False
#     return True
