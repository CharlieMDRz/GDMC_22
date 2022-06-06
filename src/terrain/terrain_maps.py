import logging

from gdpc import worldLoader

from generation.structure import AREA_STRUCTURE
import pathfinding
from terrain import EntityManager
from terrain.biomes import BiomeMap
from terrain.fluid_map import FluidMap
from terrain.height_map import HeightMap
from terrain.tree_map import TreesMap
from utils import BuildArea, BoundingBox, Position, dump, log_exec_time


class TerrainMaps:
    """
    The Map class gather all the maps representing the Minecraft Map selected for the filter
    """

    def __init__(self, level: worldLoader.WorldSlice, area: BuildArea):
        if area.width < level.heightmaps["WORLD_SURFACE"].shape[0]:
            for k, hm in level.heightmaps.items():
                level.heightmaps[k] = hm[:-1, :-1]
        self.level = level
        self.area: BuildArea = area
        from time import time
        t0 = t1 = time()
        self.height_map = HeightMap(level, area)
        log_exec_time(t1, "Computing height map")

        t1 = time()
        self.biome = BiomeMap(level, area)
        log_exec_time(t1, "Computing biome map")

        t1 = time()
        self.fluid_map = FluidMap(level, area, self)
        log_exec_time(t1, "Computing fluid map")

        self.road_network = pathfinding.road_network.RoadNetwork(self.width, self.length, self)
        self.rail_network = pathfinding.rail_network.RailNetwork(self.width, self.length, self)

        t1 = time()
        self.trees = TreesMap(level, self.height_map)
        log_exec_time(t1, "Computing forest map")

        self.entities: EntityManager = EntityManager.from_world_slice(level)
        log_exec_time(t0, "Computing terrain map")

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
        Undo all modifications to the terrain for debug purposes. Assuming all blocks are set through AREA_STRUCTURE.set
        iterates through altered positions in that structure
        """
        dump()  # clear buffer
        for x, y, z in AREA_STRUCTURE.altered_positions:
            pos: Position = Position(x, z, y)
            AREA_STRUCTURE.set(pos, self.level.getBlockAt(*pos.abs_xyz), 100000)
        dump()  # finalize reset

        self.entities.reset()
