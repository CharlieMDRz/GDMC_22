from random import randint
from typing import Tuple

import numpy as np

import terrain
from building_seeding.building_encyclopedia import BUILDING_ENCYCLOPEDIA
from building_seeding.building_pool import BuildingType
from generation.generators import Generator
from parameters import *
from terrain import ObstacleMap, TerrainMaps
from utils import *

ENTRY_POINT_MARGIN = (MIN_PARCEL_SIZE + MAX_ROAD_WIDTH) // 2


class Parcel(TransformBox):
    max_surfaces = BUILDING_ENCYCLOPEDIA["Flat_scenario"]["MaxSurface"]

    def __init__(self, seed, building_type, mc_map, width=MIN_PARCEL_SIZE, length=MIN_PARCEL_SIZE):
        # type: (Point, BuildingType, terrain.TerrainMaps, int, int) -> Parcel

        # build parcel box
        margin = AVERAGE_PARCEL_SIZE // 2
        shifted_x: int = max(margin, pos_bound(seed.x - margin, mc_map.width - margin))
        shifted_z: int = max(margin, pos_bound(seed.z - margin, mc_map.length - margin))
        self._center = Position(shifted_x + margin, shifted_z + margin)

        # init bounds
        super().__init__((self.center.x - (width // 2), 0, self.center.z - (length // 2)), (width, 1, length))
        self._mask = np.full((self.width, self.length), True)

        self.building_type: BuildingType = building_type
        self._map = mc_map  # type: terrain.TerrainMaps
        self._entry_point: Position = seed.asPosition
        if self._entry_point is not BuildingType.ghost:
            self.compute_entry_point()
        self._obstacle = None

    def __eq__(self, other):
        if not isinstance(other, Parcel):
            return False
        return self.center == other.center and self.building_type == other.building_type

    def __str__(self):
        return "{} parcel at {}".format(self.building_type.name, self.center)

    def compute_entry_point(self) -> None:
        """
        Computes entry point of the parcel based on the closest road point
        """
        road_net = self._map.road_network

        def key(road_block):
            """
            arbitrary key to find closest road point
            :param road_block:
            :return:
            """
            road_y = self._map.height_map[road_block.x, road_block.z]
            self_y = self._map.height_map[self.mean_x, self.mean_z]
            return manhattan(Point(self.mean_x, self.mean_z, self_y), Point(road_block.x, road_block.z, road_y))

        self._entry_point = argmin(road_net.road_blocks, key=key) if road_net.road_blocks else Position(0, 0)

    def expand(self, direction: Direction, **kwargs):
        """
        Extend the parcel in the given Direction
        """
        assert self.is_expendable(direction)  # trust the user
        ObstacleMap().hide_obstacle(*self.obstacle(forget=True), False)
        super().expand(direction, inplace=True)
        self._mask = np.full((self.width, self.length), True)
        # mark parcel points on obstacle terrain
        ObstacleMap().add_obstacle(*self.obstacle())

    def is_expendable(self, direction: Direction) -> bool:
        """
        Computes whether the parcel can be extended in a given Direction
        """
        if self._map is None:
            return False
        # try:
        expanded = super().expand(direction)  # expanded parcel
        obstacle = ObstacleMap()  # type: terrain.ObstacleMap  # obstacle terrain
        ext = expanded - self  # extended part of the expanded parcel

        if ext.minx < 0 or ext.minz < 0 or ext.maxx >= obstacle.width or ext.maxz >= obstacle.length:
            return False

        no_obstacle = (obstacle[ext.minx:ext.maxx, ext.minz:ext.maxz] == 0).all()
        # h = self._map.height_map.box_height(expanded, True)
        # flat_extend = (h.max() - h.min()) / min(expanded.width, expanded.length) <= 0.7
        flat_extend = True
        valid_sizes = expanded.surface <= self.max_surfaces[self.building_type.name]
        expanded_ratio = min(expanded.width / expanded.length, expanded.length / expanded.width)
        valid_ratio = (expanded.width <= 3 and expanded.length <= 3) or MIN_RATIO_SIDE <= expanded_ratio
        return no_obstacle and valid_sizes and valid_ratio and flat_extend

    def obstacle(self, margin=0, forget=False) -> Tuple[Position, np.ndarray]:
        """
        Returns the obstacle associated to the parcel
        :param margin: optional extension of the obstacle mask in every direction
        :param forget: whether to forget the previously computed mask
        :return:
        """

        if self._obstacle is None:
            if margin > 0 and self.mask.all():
                point = self.position - Point(margin, margin)
                mask = np.full((self.width + 2 * margin, self.length + 2 * margin), True)
            else:
                point = self.position
                mask = np.copy(self._mask)
            mask[0, :] = False
            mask[:, 0] = False
            self._obstacle = point, mask

        if forget:
            obs = self._obstacle
            self._obstacle = None
            return obs
        return self._obstacle

    @property
    def entry_x(self):
        return self.entry_point.x

    @property
    def entry_z(self):
        return self.entry_point.z

    @property
    def position(self):
        return Position(self.minx, self.minz)

    @property
    def mean_x(self):
        return self._center.x

    @property
    def mean_z(self):
        return self._center.z

    @property
    def generator(self):
        if self.building_type.value is None:
            return None
        return self.building_type.value(self.box, entry_point=self._entry_point, mask=self.mask)

    @property
    def height_map(self):
        return self._map.height_map.box_height(self, True)

    @property
    def center(self):
        return self._center

    @property
    def absolute_mean(self):
        return Point(self.box.minx + self.width // 2, self.box.minz + self.length // 2)

    @property
    def entry_point(self):
        return self._entry_point

    @property
    def mask(self):
        return self._mask

    @property
    def box(self):
        from utils import BuildArea
        return self.translate(BuildArea().x, 0, BuildArea().z)

    def set_height(self, y, h):
        h = min(h, max(self.width, self.length))
        self.translate(dy=y - self.miny, inplace=True)
        for _ in range(h - 1):
            super().expand(Direction.Top, inplace=True)

    def biome(self, level: TerrainMaps):
        x = randint(self.minx, self.maxx - 1)
        z = randint(self.minz, self.maxz - 1)
        return level.biome.getBiome(level.biome[x, z])


class MaskedParcel(Parcel):
    __expendable: bool

    def __init__(self, origin, building_type, mc_map=None, mask=None):
        # type: (Point, BuildingType, TerrainMaps, array) -> None

        if mask is not None:
            seed = origin + Point(mask.shape[0] // 2, mask.shape[1] // 2) if mask is not None else origin
            Parcel.__init__(self, seed, building_type, mc_map, mask.shape[0], mask.shape[1])
            # for these parcels, relative box is immutable
            self._mask = mask
            self.__expendable = False
        else:
            seed = origin + Point(MIN_PARCEL_SIZE // 2, MIN_PARCEL_SIZE // 2) if mask is not None else origin
            Parcel.__init__(self, seed, building_type, mc_map)
            self.__expendable = True

    def __valid_extended_point(self, x, z, direction: Direction):
        """Can we extend the parcel to the point x, z from a given direction"""
        obstacle: ObstacleMap = ObstacleMap()
        point = Point(x, z)
        source = point - direction.value - self.origin  # type: Point
        if not (self._mask[source.xz] and obstacle.is_accessible(point)):
            return False

        if self._map.height_map.steepness(point, norm=True) >= 1:
            return False

        return True

    def is_expendable(self, direction: Direction) -> bool:
        if not self.__expendable:
            return False
        elif super().is_expendable(direction):
            return True
        else:
            obstacle = ObstacleMap()  # type: terrain.ObstacleMap  # obstacle terrain
            expanded = TransformBox.expand(self, direction)  # expanded parcel
            ext = expanded - self  # extended part of the expanded parcel

            out_limits = ext.minx < 0 or ext.minz < 0 or ext.maxx >= obstacle.width or ext.maxz >= obstacle.length
            if out_limits:
                return False
            if not expanded.surface <= self.max_surfaces[self.building_type.name]: return False
            expanded_ratio = min(expanded.width / expanded.length, expanded.length / expanded.width)
            if not ((expanded.width <= 3 and expanded.length <= 3) or MIN_RATIO_SIDE <= expanded_ratio):
                return False

            validity = [self.__valid_extended_point(x, z, direction) for x, y, z in ext.positions]
            return sum(validity) >= len(validity) // 2

    def expand(self, direction: Direction, **kwargs) -> None:
        from numpy import insert
        index = {Direction.North: 0, Direction.East: self.width, Direction.South: self.length, Direction.West: 0}
        axis = {Direction.North: 1, Direction.East: 0, Direction.South: 1, Direction.West: 0}
        ObstacleMap().hide_obstacle(*self.obstacle(forget=True), False)

        # compute extended mask and mark it on the obstacle map
        ext = TransformBox.expand(self, direction) - self
        mask_extension = [self.__valid_extended_point(x, z, direction) for x, y, z in ext.positions]

        super(Parcel, self).expand(direction, inplace=True)
        self._mask = insert(self._mask, index[direction], np.array(mask_extension), axis=axis[direction])
        ObstacleMap().add_obstacle(*self.obstacle())

    def obstacle(self, margin=0, forget=False):
        if self.__expendable:
            return super().obstacle(margin, forget)
        mask = np.copy(self._mask)
        mask[0, :] = False
        mask[:, 0] = False
        self._obstacle = self.position, mask
        return self._obstacle

    @property
    def generator(self) -> Generator:
        return self.building_type.value(self.box, entry_point=self.entry_point, mask=self.mask)

    def add_mask(self, new_mask):
        assert self._mask.shape == new_mask.shape
        self._mask &= new_mask
