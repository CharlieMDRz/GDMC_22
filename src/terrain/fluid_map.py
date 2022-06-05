from __future__ import print_function

from itertools import product
from time import time
from typing import List

import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from gdpc.worldLoader import WorldSlice
from utils import Point, water_blocks, lava_blocks, \
    BuildArea, getBlockRelativeAt, PointArray, Direction, log_exec_time
import parameters
from terrain.biomes import BiomeMap
from utils.algorithms.fast_dijkstra import fast_dijkstra
from utils.parameters import MIN_DIST_TO_OCEAN, MIN_DIST_TO_RIVER, \
    MIN_DIST_TO_LAVA


class FluidMap(PointArray):

    def __new__(cls, level: WorldSlice, area: BuildArea, terrain, **kwargs):
        values = np.zeros((area.width, area.length), dtype=np.int0)
        obj = super().__new__(cls, values)
        obj.area = area
        obj.__water_map = np.full((area.width, area.length), 0)
        obj.__lava_map = np.full((area.width, area.length), False)
        obj.terrain = terrain
        obj.__water_limit = kwargs.get("water_limit", parameters.MAX_WATER_EXPLORATION)
        obj.__lava_limit = kwargs.get("lava_limit", parameters.MAX_LAVA_EXPLORATION)
        
        obj.__borderpts = []  # type: List[Point]
        obj.__coastline = []  # type: List[Point]

        obj.has_lava = obj.has_river = obj.has_ocean = False
        obj.detect_sources(level)

        return obj

    def detect_sources(self, level, algorithm='spread', kernel='knn', param=16):
        # type: (WorldSlice, str, str, int) -> None
        water_points = []
        t0 = time()
        for x, z in product(range(self.area.width), range(self.area.length)):
            y: int = self.terrain.height_map[x, z]
            if getBlockRelativeAt(level, x, y, z).split(':')[-1] in water_blocks:
                biome = BiomeMap.getBiome(self.terrain.biome[x, z])
                if 'ocean' in biome or 'beach' in biome:
                    label = 1
                elif 'river' in biome:
                    label = 2
                elif 'swamp' in biome:
                    label = 3
                else:
                    label = -1  # yet unlabeled
                water_points.append((x, z, label))

            elif getBlockRelativeAt(level, x, y, z) in lava_blocks:
                self.__lava_map[x, z] = True
                self.has_lava = True

        if water_points:
            data = [entry[:2] for entry in water_points]
            lbls = [entry[2] for entry in water_points]

            if algorithm in ['prop', 'spread']:
                algo = LabelPropagation if algorithm == 'prop' else LabelSpreading
                model = algo(kernel, gamma=param, n_neighbors=min(len(lbls), param), tol=1e-4, max_iter=200)
                try:
                    model.fit(data, lbls)
                    lbls = model.predict(data)
                except ValueError:
                    # no water or no labeled water point (ponds only)
                    lbls = [3 for _ in data]

            lbls = [2 if _ == -1 else _ for _ in lbls]

            for (x, z), water_type in zip(data, lbls):
                self.__water_map[x, z] = water_type
                if water_type == 1:
                    self.has_ocean = True
                elif water_type in [2, 3]:
                    self.has_river = True

        log_exec_time(t0, "Detecting water sources")
        t0 = time()
        self.__build_distance_maps()
        log_exec_time(t0, "Computing fluid distance maps")

    def __build_distance_maps(self):
        self.river_distance = np.full(self.__water_map.shape, self.__water_limit, dtype=np.float32)
        self.ocean_distance = np.full(self.__water_map.shape, self.__water_limit, dtype=np.float32)
        self.lava_distance = np.full(self.__lava_map.shape, self.__lava_limit, dtype=np.float32)

        for x, z in product(range(self.area.width), range(self.area.length)):
            if self.__water_map[x, z] in [2, 3]:
                self.river_distance[x, z] = 0
            elif self.__water_map[x, z] == 1:
                self.ocean_distance[x, z] = 0
            elif self.__lava_map[x, z]:
                self.lava_distance[x, z] = 0
            elif (x == 0) or (z == 0) or (x == (self.area.width - 1)) or (z == (self.area.length - 1)):
                self.__borderpts.append(Point(x, z))

        def is_init_neigh(distance_map, _x, _z):
            # Context: find border water points in a water surface.
            # Surrounded water points are useless in exploration
            W, L = distance_map.shape
            if distance_map[_x, _z] != 0:
                return False
            else:
                for direction in Direction.cardinal_directions():
                    x0, z0 = _x + direction.x, _z + direction.z
                    try:
                        if distance_map[x0, z0] != 0:
                            return True  # (x, z) is a water (or lava) point with a non water neighbour
                    except IndexError:
                        continue
                return False

        self.__coastline = [Point(_x, _z) for _x, _z in product(range(self.area.width), range(self.area.length))
                            if is_init_neigh(self.ocean_distance, _x, _z)]

        fast_dijkstra(self.river_distance, self.terrain.height_map[:])
        fast_dijkstra(self.ocean_distance, self.terrain.height_map[:])
        fast_dijkstra(self.lava_distance, self.terrain.height_map[:])

        # __pseudo_dijkstra(self.river_distance)
        # __pseudo_dijkstra(self.ocean_distance)
        # __pseudo_dijkstra(self.lava_distance)

        # print(self.river_distance[0, 0])
        # from matplotlib import pyplot as plt
        # plt.imshow(self.river_distance)
        # plt.show()

    def is_lava(self, x_or_point, z=None, margin=0):
        # type: (Point or int, None or int, float) -> object
        if isinstance(x_or_point, Point):
            p = x_or_point
            return self.lava_distance[p.x, p.z] <= margin
        else:
            x = x_or_point
            return self.lava_distance[x, z] <= margin

    def is_close_to_fluid(self, x_or_point, z=None):
        # type: (Point or int, None or int) -> object
        if isinstance(x_or_point, Point):
            p = x_or_point
            return self.is_close_to_fluid(p.x, p.z)
        else:
            x = x_or_point
            return ((self.ocean_distance[x][z] <= MIN_DIST_TO_OCEAN)
                    | (self.river_distance[x][z] <= MIN_DIST_TO_RIVER)
                    | (self.lava_distance[x][z] <= MIN_DIST_TO_LAVA))

    def is_water(self, x_or_point, z=None, margin=0):
        # type: (int or Point, int or None, float) -> bool
        if isinstance(x_or_point, Point):
            return self.is_water(x_or_point.x, x_or_point.z)
        else:
            x = x_or_point
            return (self.ocean_distance[x, z] <= margin) | (self.river_distance[x, z] <= margin)

    @property
    def as_obstacle_array(self):
        # if one of the conditions is valid, fluids are an obstacle
        obs = ((self.ocean_distance <= MIN_DIST_TO_OCEAN)
               | (self.river_distance <= MIN_DIST_TO_RIVER)
               | (self.lava_distance <= MIN_DIST_TO_LAVA))  # type: np.ndarray
        return obs.astype(int)

    @property
    def water(self):
        return self.__water_map

    @property
    def external_connections(self):
        return self.__borderpts + self.__coastline

    def water_distance(self, px, pz=None):
        if pz is None:
            return self.water_distance(px.x, px.z)
        px, pz = int(round(px)), int(round(pz))
        res = self.width * self.length
        if self.has_ocean: res = min(res, self.ocean_distance[px, pz])
        if self.has_river: res = min(res, self.river_distance[px, pz])
        return int(res)
