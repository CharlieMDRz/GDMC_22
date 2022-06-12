import random
from collections import Counter
from typing import List

from building_seeding.district.district_builders import DistrictCluster
from generation import HousePalette
from parameters import PALETTE_MUTATION_PROBABILITY
from terrain import TerrainMaps
from utils import Position, bernouilli


class Town:
    def __init__(self, center: Position, name: str, palette):
        self.__name: str = name
        self.__center: Position = center
        self.__palettes: List[HousePalette] = [palette]

    @classmethod
    def fromCluster(cls, dc: DistrictCluster, terrain: TerrainMaps):
        from generation.building_palette import get_biome_palette
        center = dc.center
        name = cls.genName()

        biome_occurrence = Counter(map(lambda pos: terrain.biome[pos], dc.reps))
        town_biome: int = biome_occurrence.most_common()[0][0]
        biome_name: str = terrain.biome.getBiome(town_biome)
        palette = get_biome_palette(biome_name)

        return cls(center, name, palette)

    @property
    def center(self) -> Position:
        return self.__center

    @property
    def palette(self):
        palette = random.choice(self.__palettes)
        if bernouilli(PALETTE_MUTATION_PROBABILITY):
            palette = HousePalette.mutant(palette)
            self.__palettes.append(palette)
        return palette

    @property
    def name(self):
        return self.__name

    @staticmethod
    def genName() -> str:
        from building_seeding.district.districts import CityNameGenerator
        return CityNameGenerator().generate()

    @staticmethod
    def genPalette(terrain, districtMap) -> str:
        pass
