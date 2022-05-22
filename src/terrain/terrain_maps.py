import logging

from gdpc import worldLoader

from generation.structure import AREA_STRUCTURE
from terrain import EntityManager
from terrain.biomes import BiomeMap
from terrain.fluid_map import FluidMap
from terrain.height_map import HeightMap
from terrain.tree_map import TreesMap
from utils import BuildArea, BoundingBox, Position, dump


class TerrainMaps:
    """
    The Map class gather all the maps representing the Minecraft Map selected for the filter
    """

    def __init__(self, level: worldLoader.WorldSlice, area: BuildArea):
        from path_networks import RoadNetwork, RailNetwork
        if area.width < level.heightmaps["WORLD_SURFACE"].shape[0]:
            for k, hm in level.heightmaps.items():
                level.heightmaps[k] = hm[:-1, :-1]
        self.level = level
        self.area: BuildArea = area
        from time import time
        t0 = t1 = time()
        self.height_map = HeightMap(level, area)
        print(f'Computed height map in {time() - t1}')

        t1 = time()
        self.biome = BiomeMap(level, area)
        print(f'Computed biome map in {time() - t1}')

        t1 = time()
        self.fluid_map = FluidMap(level, area, self)
        print(f'Computed fluid map in {time() - t1}')

        t1 = time()
        self.road_network = RoadNetwork(self.width, self.length, self)  # type: RoadNetwork
        self.rail_network: RailNetwork = RailNetwork(self.width, self.length, self)
        print(f'Computed road map in {time() - t1}')

        t1 = time()
        self.trees = TreesMap(level, self.height_map)
        print(f'Computed trees map in {time() - t1}')

        t1 = time()
        print(f'Computed terrain maps in {t1 - t0}')

        self.entities: EntityManager = EntityManager.from_world_slice(level)

    @property
    def width(self):
        return self.area.width

    @property
    def length(self):
        return self.area.length

    @property
    def shape(self):
        return self.width, self.length

    @property
    def box(self):
        return BoundingBox((self.area.x, 0, self.area.z), (self.width, 256, self.length))

    def in_limits(self, point, absolute_coords):
        if absolute_coords:
            return point in self.area
        else:
            return point + self.area.origin in self.area

    @staticmethod
    def request(build_area_json=None):
        from time import time
        logging.info("Requesting build area...")
        area = BuildArea(build_area_json)
        logging.info(f"Found {str(area)}")
        print("Requesting level...")
        t0 = time()
        level = worldLoader.WorldSlice(area.x, area.z, area.x + area.width, area.z + area.length)
        print(f"completed in {(time() - t0)}s")
        return TerrainMaps(level, area)

    def undo(self):
        """
        Undo all modifications to the terrain for debug purposes
        """
        dump()
        current_terrain = TerrainMaps.request(self.area.json)
        old_level = self.level
        new_level = current_terrain.level
        for pos in BuildArea.building_positions():  # type: Position
            min_y = min(self.height_map.lower_height(pos.x, pos.z),
                        current_terrain.height_map.lower_height(pos.x, pos.z))
            max_y = max(self.height_map.upper_height(pos.x, pos.z),
                        current_terrain.height_map.upper_height(pos.x, pos.z))
            for y in range(min_y - 2, max_y + 2):
                coords = pos.abs_x, y, pos.abs_z
                if old_level.getBlockAt(*coords) != new_level.getBlockAt(*coords):
                    AREA_STRUCTURE.set(pos.withCoords(y=y), old_level.getBlockAt(*coords), 100000)
        dump()

        self.entities.reset()
