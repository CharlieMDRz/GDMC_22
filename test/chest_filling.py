import os

import gdpc.toolbox

from generation.structure import AREA_STRUCTURE
from terrain import TerrainMaps
from utils import Position
from utils.loot_table_sampler import LootTable

if __name__ == '__main__':
    loot_tables_dir = 'resources/data_1.16.5/loot_tables/chests/village'
    terrain = TerrainMaps.request()

    mean_pos = Position(terrain.width//2, terrain.length//2)
    chest_pos = mean_pos.withCoords(y=terrain.height_map[mean_pos.xz])

    for loot_tables_path in os.listdir(loot_tables_dir):
        loot_tables_path = os.path.join(loot_tables_dir, loot_tables_path)
        print(loot_tables_path)
        if 'armour' not in loot_tables_path:
            continue

        chest_pos += (0, 1, 0)

        loot_table = LootTable.fromMCLootTable(loot_tables_path)
        AREA_STRUCTURE.set(chest_pos, 'chest')
        AREA_STRUCTURE.dump()
        gdpc.toolbox.placeInventoryBlock(*chest_pos.abs_xyz, items=loot_table.sample(9, 3))

    input('enter to undo')
    terrain.undo()
