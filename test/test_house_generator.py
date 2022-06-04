from random import choice, randint
from time import sleep

from generation import ProcHouseGenerator
from generation.building_palette import biome_palettes, HousePalette, random_palette

from terrain import TerrainMaps
from utils import TransformBox, dump

displayName = "House generator test filter"

N_HOUSES = 1

if __name__ == '__main__':
    terrain = TerrainMaps.request()
    x, z = terrain.area.x, terrain.area.z
    w, l = terrain.area.width, terrain.area.length
    y = terrain.height_map[0, 0]

    all_palettes = []
    for palettes in biome_palettes.values():
        if isinstance(palettes, HousePalette):
            all_palettes.append(palettes)
        else:
            all_palettes.extend(palettes)
    for _ in range(N_HOUSES):
        terrain.undo()
        box = TransformBox((x, y, z), (w, randint(4, 16), l))
        ProcHouseGenerator(box).generate(terrain, terrain.height_map.box_height(box, False), random_palette())
        dump()  # finalize test gen
        sleep(4)
    terrain.undo()
