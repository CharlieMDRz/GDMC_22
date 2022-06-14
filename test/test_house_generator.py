import traceback
from random import choice, randint
from time import sleep

from generation import ProcHouseGenerator
from generation.building_palette import biome_palettes, HousePalette, random_palette, PaletteGenerators, \
    HousePaletteGenerator
from generation.structure import AREA_STRUCTURE

from terrain import TerrainMaps
from utils import TransformBox, dump, BlockAPI

materials = BlockAPI.blocks

displayName = "House generator test filter"

N_HOUSES = 1


def build_house(box, terrain, palette, time=None):
    AREA_STRUCTURE.reset()
    ProcHouseGenerator(box).generate(terrain, terrain.height_map.box_height(box, False), palette)
    dump()  # finalize test gen
    if time:
        sleep(time)
    else:
        input("Enter anything to remove house")
    terrain.undo()


if __name__ == '__main__':
    terrain = TerrainMaps.request()
    x, z = terrain.area.x+1, terrain.area.z+1
    w, l = terrain.area.width-2, terrain.area.length-2
    y = terrain.height_map[0, 0]
    box = TransformBox((x, y, z), (w, randint(4, 16), l))

    # all_palettes = []
    # for palettes in biome_palettes.values():
    #     if isinstance(palettes, HousePalette):
    #         all_palettes.append(palettes)
    #     else:
    #         all_palettes.extend(palettes)
    # for _ in range(N_HOUSES):
    #     build_house(box, terrain, random_palette())
    # terrain.undo()
    #
    # palette = HousePalette(materials.Stone,
    #                        materials.BlackGlazedTerracotta,
    #                        materials.SmoothSandstone,
    #                        materials.BirchPlanks,
    #                        materials.WhiteStainedGlassPane,
    #                        'gable', materials.PolishedBlackstoneBrickStairs,
    #                        'birch', materials.SmoothSandstone,
    #                        materials.ChiseledSandstone)
    palette = PaletteGenerators.PARIS()

    try:
        build_house(box, terrain, palette)
    except Exception:
        traceback.print_exc()
    finally:
        terrain.undo()

# for city_palette_name in vars(PaletteGenerators):
#     city_palette = getattr(PaletteGenerators, city_palette_name)
#     if isinstance(city_palette, HousePaletteGenerator):
#         build_house(TransformBox((x, y, z), (w, randint(6, 14), l)), terrain, city_palette(), 5)
