from terrain import TerrainMaps, ObstacleMap
from terrain.structure_detection import StructureDetector
from utils import dump

if __name__ == '__main__':
    terrain = TerrainMaps.request()
    ObstacleMap.from_terrain(terrain)
    for parcel in StructureDetector(terrain).get_structure_parcels():
        parcel.generator.generate(terrain, terrain.height_map.box_height(parcel.box, False))
    dump()  # finalize test gen

    input("Undo ?")
    terrain.undo()
