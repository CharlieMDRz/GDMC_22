from generation import WindmillGenerator
from generation.structure import AREA_STRUCTURE
from terrain import TerrainMaps, ObstacleMap
from utils import TransformBox, Position, Direction


def generate_windmill(terrain):
    x, z = terrain.area.x, terrain.area.z
    w, l = terrain.area.width, terrain.area.length
    y = terrain.height_map[0, 0]
    box = TransformBox((x, y, z), (w, 1, l))
    WindmillGenerator(box).generate(terrain, terrain.height_map)


def generate_train_station(terrain, direction: Direction):
    ObstacleMap.from_terrain(terrain)
    rail_x = terrain.width // 2
    rail_z = terrain.length // 2
    rail_y = terrain.height_map[rail_x, terrain.length // 2] + 5
    if direction in [Direction.North, Direction.South]:
        rails = [Position(rail_x, z, rail_y) for z in range(1, terrain.length-1)]
        rail_edges = [rails[0], rails[-1]]
        terrain.rail_network.create_road(*rail_edges, path=rails)
        terrain.rail_network.add_station(rails[len(rails)//2], Direction.North)
    else:
        rails = [Position(x, rail_z, rail_y) for x in range(1, terrain.width-1)]
        rail_edges = [rails[0], rails[-1]]
        terrain.rail_network.create_road(*rail_edges, path=rails)
        terrain.rail_network.add_station(rails[len(rails)//2], Direction.West)
    terrain.rail_network.generate(terrain, None)


if __name__ == '__main__':
    terrain = TerrainMaps.request()
    structure = 'train_station'
    if structure == 'windmill':
        generate_windmill(terrain)
    elif structure == 'train_station':
        generate_train_station(terrain, Direction.East)
    if not input():
        terrain.undo()
