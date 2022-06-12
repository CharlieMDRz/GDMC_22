from terrain import TerrainMaps, HeightMap
from utils import Position


def targeted_block_at(terrain: TerrainMaps, pos: Position, targets):
    hm: HeightMap = terrain.height_map
    y = hm.upper_height(pos)
    block_state: str = terrain.level.getBlockAt(pos.abs_x, y, pos.abs_z)
    return any(block_state.replace("minecraft:", '').startswith(bs) for bs in targets)


def locate_at_surface(terrain: TerrainMaps, blocks_to_locate):
    locations = []
    for position in terrain.area.building_positions():
        if targeted_block_at(terrain, position, blocks_to_locate):
            locations.append(position)
    return locations
