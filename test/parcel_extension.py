from building_seeding import MaskedParcel, BuildingType
from generation import CropGenerator
from generation.building_palette import oak_palette1
from settlement import Settlement
from terrain import TerrainMaps, ObstacleMap
from utils_test import locate_at_surface
from utils import BlockAPI

if __name__ == '__main__':
    terrain = TerrainMaps.request()
    settlement = Settlement(terrain)

    seeds = locate_at_surface(terrain, [BlockAPI.blocks.DiamondBlock])
    obs = ObstacleMap.from_terrain(terrain)
    settlement._parcels.extend(MaskedParcel(seed, BuildingType.crop, terrain) for seed in seeds)
    settlement.define_parcels()
    for parcel in settlement._parcels:
        gen: CropGenerator = parcel.generator
        gen._gen_animal_farm(parcel.height_map, oak_palette1, entities=terrain.entities)

    input("undo ?")
    terrain.undo()
