import random
import time
from typing import List, Dict

from building_seeding import InterestSeeder, BuildingType, VillageSkeleton, Parcel
from geometry import line2d
from settlement import Settlement, dump
from terrain import TerrainMaps, ObstacleMap
from utils import detect_entities, Entity, Position, setBlock, Point

if __name__ == '__main__':
    terrain: TerrainMaps = TerrainMaps.request()
    ObstacleMap.from_terrain(terrain)
    settlement = Settlement(terrain)
    settlement.build_districts(n_clusters=1)
    parcels: List[Parcel] = settlement._parcels
    parcel_for_type: Dict[str, Parcel] = {}
    interest = InterestSeeder(terrain, settlement.districts, parcels, "Flat_scenario")
    skeleton = VillageSkeleton("Flat_scenario", terrain, settlement.districts, parcels)
    btype = BuildingType.crop

    entities: List[Entity] = detect_entities(terrain.level)
    e_types = {_.entity_type for _ in entities}

    for e_type in e_types:
        parcel_pos = interest.get_seed(btype)
        skeleton.add_parcel(parcel_pos, btype)
        parcel_for_type[e_type] = parcels[-1]

    settlement.define_parcels()
    for parcel in parcels:
        gen = parcel.generator
        gen._clear_trees(terrain)
        x0, _, z0 = gen.origin
        x1 = x0 + gen.width - 1
        z1 = z0 + gen.length - 1
        for line in ((x0, z0, x0, z1), (x1, z0, x1, z1), (x0, z0, x1, z0), (x0, z1, x1, z1)):
            for x, z in line2d(*line):
                y = terrain.height_map[x - terrain.area.x, z - terrain.area.z] + 1
                for dy in range(y, y+2):
                    setBlock(Point(x, z, dy), "oak_fence")
    dump()  # finalize fence

    for entity in entities:
        parcel = parcel_for_type[entity.entity_type]
        try:
            x = random.randint(parcel.origin.x + 1, parcel.origin.x + parcel.width - 2)
            z = random.randint(parcel.origin.z + 1, parcel.origin.z + parcel.length - 2)
        except ValueError:
            continue
        y = terrain.height_map[x, z] + 1
        pos: Position = Position(x, z, y)
        entity.move_to((pos.abs_x, y, pos.abs_z))
        time.sleep(.1)

    if input():
        terrain.undo()
