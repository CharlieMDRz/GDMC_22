from typing import List

import numpy as np
import cv2

from gdpc.worldLoader import WorldSlice
from utils import BuildArea, ground_blocks, water_blocks, lava_blocks, getBlockRelativeAt, PointArray
from utils import Point


class HeightMap(PointArray):
    __air_height: PointArray

    def __new__(cls, level: WorldSlice, area: BuildArea):
        # highest non air block
        obj = super().__new__(cls, HeightMap.__calcGoodHeightmap(level))
        obj.area = area
        obj.__air_height = PointArray(level.heightmaps["WORLD_SURFACE"][:] - 1)

        # highest solid block (below oceans)
        obj.__ocean_floor: PointArray = PointArray(np.minimum(obj[:], level.heightmaps["OCEAN_FLOOR"]))

        # uses absolute coordinates
        obj.__origin = Point(area.x, area.z)

        steepness_z = cv2.Scharr(obj.astype(np.uint8), 5, 1, 0) / (2 ** 5)  # normalizer
        steepness_x = cv2.Scharr(obj.astype(np.uint8), 5, 0, 1) / (2 ** 5)

        obj.__steepness_x = cv2.GaussianBlur(steepness_x, (15, 15), 0)
        obj.__steepness_z = cv2.GaussianBlur(steepness_z, (15, 15), 0)

        return obj

    def upper_height(self, xr: Point or int, zr=None):
        """
        :param xr: X coordinate or Point instance
        :param zr: Z coordinate or None
        :return: Y coordinate of the highest non air block
        """
        if zr is None:
            return self.__air_height[xr]
        return self.__air_height[xr, zr]

    def lower_height(self, xr, zr=None):
        """
        :param xr: X coordinate or Point instance
        :param zr: Z coordinate or None
        :return: Y coordinate of the highest ground block (below oceans, and trees)
        """
        if zr is None:
            return self.__ocean_floor[xr]
        return self.__ocean_floor[xr, zr]

    def box_height(self, box, use_relative_coords, include_fluids=False):
        x0 = box.minx if use_relative_coords else box.minx - self.__origin.x
        z0 = box.minz if use_relative_coords else box.minz - self.__origin.z
        matrix = self[:] if include_fluids else self.__ocean_floor[:]
        return matrix[x0: (x0 + box.width), z0:(z0 + box.length)].astype(int)

    def steepness(self, x, z=None, norm=True):
        if isinstance(x, Point):
            return self.steepness(x.x, x.z, norm)
        steepness_vector = Point(self.__steepness_x[x, z], self.__steepness_z[x, z])
        return steepness_vector.norm if norm else steepness_vector

    def update(self, points, heights):
        # type: (List[Point], List[int]) -> None
        for p, h in zip(points, heights):
            if self.upper_height(p) == self[p]:
                self.__air_height[p] = h
            if self.lower_height(p) == self[p]:
                self.__ocean_floor[p] = h
            self[p] = h

    @staticmethod
    def __calcGoodHeightmap(world_slice):
        """Calculates a heightmap that is well suited for building.
        It ignores any logs and leaves and treats water as ground.

        Args:
            world_slice (WorldSlice): an instance of the WorldSlice class containing the raw heightmaps and block data

        Returns:
            any: numpy array containing the calculated heightmap
        """
        hm_mbnl = world_slice.heightmaps["MOTION_BLOCKING_NO_LEAVES"]
        heightmapNoTrees = hm_mbnl[:]
        area = world_slice.rect

        valid_blocks = ground_blocks.union(water_blocks).union(lava_blocks)

        def isValidBlock(block):
            block = block.split(':')[-1].split('[')[0]
            return block in valid_blocks

        from itertools import product
        for x, z in product(range(area[2]), range(area[3])):
            y = heightmapNoTrees[x, z]
            while not isValidBlock(getBlockRelativeAt(world_slice, x, y-1, z)):
                y -= 1
            heightmapNoTrees[x, z] = y

        return np.array(np.minimum(hm_mbnl, heightmapNoTrees)) - 1
