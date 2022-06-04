"""
Test of the masked parcels and city block parcelling
"""

from building_seeding import VillageSkeleton, BuildingType
from building_seeding.village_skeleton import CityBlock, cycle_preprocess
from settlement import Settlement
from terrain import TerrainMaps, ObstacleMap
from utils import Point, dump

if __name__ == '__main__':
    terrain = TerrainMaps.request()
    ObstacleMap.from_terrain(terrain)
    print("Hello Settlers!")
    W, L = terrain.width, terrain.length

    # create road cycle
    p1, p2, p3, p4 = Point(int(.1 * W), int(.2 * L)), Point(int(.8 * W), int(.1 * L)), Point(int(.7 * W),
                                                                                             int(.7 * L)), Point(
        int(.3 * W), int(.9 * L))
    net = terrain.road_network
    p12 = net.create_road(p1, p2)
    p23 = net.create_road(p2, p3)
    p34 = net.create_road(p3, p4)
    p41 = net.create_road(p1, p4)
    assert len(p12) * len(p23) * len(p34) * len(p41)
    del p12, p23, p34, p41

    settlement = Settlement(terrain)
    skeleton = VillageSkeleton('Flat_scenario', terrain, settlement.districts, settlement._parcels)

    road_cycle = cycle_preprocess(net.road_blocks)
    city_block = CityBlock(road_cycle, terrain)
    for parcel in city_block.parcels(BuildingType.house):
        skeleton.add_parcel(parcel)
    settlement.define_parcels()  # define parcels around seeds
    settlement.generate()  # build buildings on parcels
    dump()  # finalize test gen

    # Optional erasing of the generated settlement
    do_undo = input("Undo ? [y]/n").lower()
    if do_undo in {"", "y"}:
        terrain.undo()
