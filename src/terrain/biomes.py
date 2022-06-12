import time
from enum import Enum
from itertools import product

import cv2
import numpy as np

from gdpc.worldLoader import WorldSlice
from utils import BuildArea, Point, Position, PointArray


class Biomes(Enum):
    Badlands = (2.0, "Dry", "badlands")
    Desert = (2.0, "Dry", "desert")
    Savanna = (1.2, "Dry", "savanna")
    ShatteredSavanna = (1.1, "Dry", "shattered_savanna")
    SavannaPlateau = (1.0, "Dry", "savanna_plateau")
    ShatteredSavannaPlateau = (1.0, "Dry", "shattered_savanna_plateau")
    Jungle = (0.95, "Temperate", "jungle")
    MushroomFields = (0.9, "Temperate", "mushroom_fields")
    Beach = (0.8, "Temperate", "beach")
    Plains = (0.8, "Temperate", "plains")
    Swamp = (0.8, "Temperate", "swamp")
    DripstoneCaves = (0.8, "Temperate", "dripstone_caves")
    DarkForest = (0.7, "Temperate", "dark_forest")
    Forest = (0.7, "Temperate", "forest")
    TallBirchForest = (0.6, "Temperate", "tall_birch_forest")
    Birchforest = (0.6, "Temperate", "birch_forest")
    Lushcaves = (0.5, "Temperate", "lush_caves")
    River = (0.5, "Temperate", "river")
    Ocean = (0.5, "Aquatic", "ocean")
    DeepFrozenOcean = (0.5, "Aquatic", "deep_frozen_ocean")
    MountainMeadow = (0.3, "Cold", "mountain_meadow")
    GiantTreeTaiga = (0.3, "Cold", "giant_tree_taiga")
    GiantSpruceTaiga = (0.25, "Cold", "giant_spruce_taiga")
    Taiga = (0.25, "Cold", "taiga")
    StoneShore = (0.2, "Cold", "stone_shore")
    ExtremeHills = (0.2, "Cold", "extreme_hills")
    SnowyBeach = (0.05, "Snowy", "snowy_beach")
    FrozenOcean = (0.0, "Snowy", "frozen_ocean")
    FrozenRiver = (0.0, "Snowy", "frozen_river")
    SnowyTundra = (0.0, "Snowy", "snowy_tundra")
    MountainGrove = (-0.2, "Snowy", "mountain_grove")
    Snowyslopes = (-0.3, "Snowy", "snowy_slopes")
    Snowytaiga = (-0.5, "Snowy", "snowy_taiga")
    Loftypeaks = (-0.7, "Snowy", "lofty_peaks")

    @classmethod
    def from_name(cls, name: str):
        biome_names = {b.name.lower(): b for b in Biomes}
        for subname in name.split('_'):
            match = list(filter(lambda b_name: subname in b_name, biome_names.keys()))
            if match: return biome_names[match[0]]
        if "mountains" in name:
            return Biomes.ExtremeHills
        elif "void" in name:
            return Biomes.Loftypeaks
        elif name == "ice_spikes":
            return Biomes.FrozenOcean
        return None

    @property
    def temperature(self):
        return self.value[0]

    @property
    def type(self):
        return self.value[1]


class BiomeMap(PointArray):

    __temperature: np.ndarray = None

    def __new__(cls, level: WorldSlice, area: BuildArea):
        # biomes
        values = np.zeros((level.chunkRect[2] * 4, level.chunkRect[3] * 4))
        for chunkX, chunkZ in product(range(level.chunkRect[2]), range(level.chunkRect[3])):
            chunkID = chunkX + level.chunkRect[2] * chunkZ
            biomes = level.nbtfile['Chunks'][chunkID]['Level']['Biomes'][:16]
            x0, z0 = chunkX * 4, chunkZ * 4
            values[x0: (x0+4), z0: (z0+4)] = np.array(biomes).reshape((4, 4), order='F')

        obj = super().__new__(cls, values)
        obj.__offset = Point(area.x % 16, area.z % 16)
        obj.awidth = level.rect[2]
        obj.alength = level.rect[3]
        return obj

    def __getitem__(self, item):
        if not isinstance(item, Point):
            return self[Point(item[0], item[1])]
        biomePoint: Point = (item + self.__offset) // 4
        # return Map.__getitem__(self, biomePoint)
        return int(super(BiomeMap, self).__getitem__(biomePoint))

    _biome_types = {
        ("ocean", 0), ("taiga", 5), ("plains", 1), ("mountains", 3), ("desert", 2), ("forest", 4), ("swamp", 6),
        ("river", 7), ("frozen_ocean", 10), ("frozen_river", 11),
        ("snowy_tundra", 12), ("snowy_mountains", 13), ("mushroom_fields", 14), ("mushroom_field_shore", 15),
        ("beach", 16), ("desert_hills", 17), ("wooded_hills", 18), ("taiga_hills", 19), ("mountain_edge", 20),
        ("jungle", 21), ("jungle_hills", 22), ("jungle_edge", 23), ("deep_ocean", 24), ("stone_shore", 25),
        ("snowy_beach", 26), ("birch_forest", 27), ("birch_forest_hills", 28), ("dark_forest", 29), ("snowy_taiga", 30),
        ("snowy_taiga_hills", 31), ("giant_tree_taiga", 32), ("giant_tree_taiga_hills", 33), ("wooded_mountains", 34),
        ("savanna", 35), ("savanna_plateau", 36), ("badlands", 37), ("wooded_badlands_plateau", 38),
        ("badlands_plateau", 39), ("warm_ocean", 44), ("lukewarm_ocean", 45), ("cold_ocean", 46),
        ("deep_warm_ocean", 47),
        ("deep_lukewarm_ocean", 48), ("deep_cold_ocean", 49), ("deep_frozen_ocean", 50), ("the_void", 127),
        ("sunflower_plains", 129), ("desert_lakes", 130), ("gravelly_mountains", 131), ("flower_forest", 132),
        ("taiga_mountains", 133), ("swamp_hills", 134), ("ice_spikes", 140), ("modified_jungle", 149),
        ("modified_jungle_edge", 151), ("tall_birch_forest", 155), ("tall_birch_hills", 156),
        ("dark_forest_hills", 157),
        ("snowy_taiga_mountains", 158), ("giant_spruce_taiga", 160), ("giant_spruce_taiga_hills", 161),
        ("modified_gravelly_mountains", 162), ("shattered_savanna", 163), ("shattered_savanna_plateau", 164),
        ("eroded_badlands", 165), ("modified_wooded_badlands_plateau", 166), ("modified_badlands_plateau", 167),
        ("bamboo_jungle", 168), ("bamboo_jungle_hills", 169)
    }

    # bi map from biome name to biome id
    __biome_to_id = {_[0]: _[1] for _ in _biome_types}
    __id_to_biome = {_[1]: _[0] for _ in _biome_types}

    @staticmethod
    def getBiome(biome_id: int):
        return BiomeMap.__id_to_biome[biome_id]

    @staticmethod
    def getBiomeId(biome_type: str):
        return BiomeMap.__biome_to_id[biome_type]

    def temperature(self, point: Position) -> float:
        biome_name = self.getBiome(self[point.xz])
        biome = Biomes.from_name(biome_name)
        return biome.temperature

    @property
    def temperature_map(self) -> np.ndarray:
        mult = 100
        if self.__temperature is None:
            t0 = time.time()
            init_temperature = np.array([[self.temperature(Position(x, z)) * mult for z in range(self.alength)] for x in range(self.awidth)], dtype=np.uint8)
            self.__temperature = cv2.blur(init_temperature, (15, 15)) / mult
            print(f"Computed temperature map in {(time.time() - t0):0.3f} seconds")

        return self.__temperature


if __name__ == '__main__':
    assert BiomeMap.getBiomeId("warm_ocean") == 44
    assert BiomeMap.getBiome(3) == "mountains"
    print({_ for _ in map(lambda _: _[0], BiomeMap._biome_types)})

    # terrain = TerrainMaps.request()
    # print(terrain.biome[0, 0])
    for biome_name, _ in BiomeMap._biome_types:
        biome = Biomes.from_name(biome_name)
        if biome is None:
            print("NO MATCH: ", biome_name)
            break
        print(biome_name, biome.name)
