import os
import random
from itertools import product
from typing import List

import gdpc.toolbox
import numpy as np
from gdpc import lookup

from generation.generators import BoundingBox
from generation.structure import AREA_STRUCTURE
from terrain import EntityManager
from utils import Position, BlockAPI, Direction, bernouilli
from utils.block_utils import build_block_state
from utils.entities import get_item_frame_entity, Entity
from utils.items import DISCS
from utils.loot_table_sampler import LootTable

wall_decorations = DISCS + [
    'clock', 'cod', 'salmon', 'book', 'pufferfish'
]


class RoomLookAround:

    @staticmethod
    def get_direction(pos):
        return next(-dir for dir in Direction.cardinal_directions(False) if not RoomLookAround.is_air(pos + dir.value))

    @staticmethod
    def is_air(pos):
        return AREA_STRUCTURE[pos] in (10, RoomFurnisher.PRIORITY)

    @staticmethod
    def is_along_window(wall_pos):
        if any(AREA_STRUCTURE[wall_pos + (dx, 0, dz)] == 7 for dx, dz in ((0, 1), (1, 0), (0, -1), (-1, 0))):
            return True
        return False


class FurniturePlacer:
    def __init__(self, block: str):
        self._block: str = block

    def place(self, position: Position):
        AREA_STRUCTURE.set(position, self._block, RoomFurnisher.PRIORITY)
        return [position]


class BedPlacer(FurniturePlacer):

    def __init__(self, block: str):
        if all(color not in block for color in lookup.COLORS):
            block = f"{random.choice(lookup.COLORS)}_bed"
        super().__init__(block)

    def place(self, position: Position):
        direction = RoomLookAround.get_direction(position)
        head_block_state = build_block_state(self._block, True, facing=(-direction).name.lower(), part='head')
        AREA_STRUCTURE.set(position, head_block_state, RoomFurnisher.PRIORITY)
        foot_block_state = build_block_state(self._block, True, facing=(-direction).name.lower(), part='foot')
        AREA_STRUCTURE.set(position+direction.value, foot_block_state, RoomFurnisher.PRIORITY)
        return [position, position + direction.value]


class CardinalPlacer(FurniturePlacer):
    BLOCKS = ['furnace']

    def place(self, position: Position):
        direction = RoomLookAround.get_direction(position)
        block_state = build_block_state(self._block, force_properties=True, facing=direction.name.lower())
        AREA_STRUCTURE.set(position, block_state, RoomFurnisher.PRIORITY)
        return [position]


class ChestPlacer(CardinalPlacer):
    def place(self, position: Position):
        super().place(position)
        AREA_STRUCTURE.dump()
        loot_tables_dir = 'resources/data_1.16.5/loot_tables/chests/village'
        loot_table_file_name = random.choice(os.listdir(loot_tables_dir))
        loot_tables_path = os.path.join(loot_tables_dir, loot_table_file_name)
        loot_table = LootTable.fromMCLootTable(loot_tables_path)
        AREA_STRUCTURE.dump()
        gdpc.toolbox.placeInventoryBlock(*position.abs_xyz, items=loot_table.sample(9, 3))
        return [position]


class FurniturePlacerFactory:

    @staticmethod
    def get_placer(block: str):
        if 'bed' in block:
            return BedPlacer(block)
        if 'chest' in block:
            return ChestPlacer(block)
        if any(_ in block for _ in CardinalPlacer.BLOCKS):
            return CardinalPlacer(block)
        return FurniturePlacer(block)


class RoomFurnisher:

    PRIORITY = 99

    def __init__(self, boxes: List[BoundingBox]):
        self.__floor_positions: List[Position] = self.detect_furnishable_floor(boxes)
        self.__wall_positions: List[Position] = self.detect_decorable_wall(self.__floor_positions)
        self.__surfaces: List[Position] = []
        self.__furniture_list = ['bed', 'chest', BlockAPI.blocks.Furnace, BlockAPI.blocks.CraftingTable, BlockAPI.blocks.Bookshelf, BlockAPI.blocks.Jukebox]

    def furnish(self):
        for furniture_piece in self.__furniture_list:
            position = random.choice(self.__floor_positions)
            furnished_positions = FurniturePlacerFactory.get_placer(furniture_piece).place(position)
            if not any(_ in furniture_piece for _ in ('bed', 'chest')):
                self.__surfaces.append(position + (0, 1, 0))
            for furnished_pos in furnished_positions:
                for furnishable_positions in (self.__wall_positions, self.__floor_positions):
                    if furnished_pos in furnishable_positions:
                        furnishable_positions.remove(furnished_pos)

        for wall_position in self.__wall_positions:
            if (wall_position - (0, 1, 0)) in self.__floor_positions:
                if bernouilli(.5):
                    self.__decorate_wall(wall_position)
            elif wall_position in self.__surfaces and bernouilli(2/3):
                self.__decorate_surface(wall_position)

    def test_furnish(self):
        for pos in self.__floor_positions:
            FurniturePlacer(BlockAPI.blocks.CyanWool).place(pos)

        for pos in self.__wall_positions:
            FurniturePlacer(BlockAPI.blocks.PurpleWool).place(pos)

    @staticmethod
    def detect_furnishable_floor(boxes: List[BoundingBox]) -> List[Position]:
        box: BoundingBox
        mega_box: BoundingBox = boxes[0]
        for box in boxes:
            mega_box = mega_box.union(box)

        valid_positions = np.full((mega_box.width, mega_box.length), False)

        for box in boxes:
            rx, rz = box.minx - mega_box.minx, box.minz - mega_box.minz
            width, length = box.width, box.length
            valid_positions[(rx+1):(rx+width-1), (rz+1):(rz+length-1)] = True
            valid_positions[(rx+2):(rx+width-2), (rz+2):(rz+length-2)] = False

        x0, y0, z0 = mega_box.origin
        positions_along_walls = {Position(x0 + dx, z0 + dz, y0, True) for dx, dz in zip(*np.where(valid_positions))}

        return [p for p in positions_along_walls if RoomFurnisher.valid_position(p)]

    @staticmethod
    def detect_decorable_wall(furnishable_floor: List[Position]) -> List[Position]:
        decorable_walls: List[Position] = []
        for floor_pos in furnishable_floor:
            wall_pos = floor_pos + (0, 1, 0)
            if not RoomLookAround.is_along_window(wall_pos):
                decorable_walls.append(wall_pos)

        return decorable_walls

    @staticmethod
    def valid_position(pos):
        if AREA_STRUCTURE[pos - (0, 1, 0)] not in (11, 15):
            return False  # over stair case
        if not all(RoomLookAround.is_air(pos + (0, i, 0)) for i in range(3)):
            return False  # under stair case
        if any(AREA_STRUCTURE[pos + (dx, 0, dz)] in (13, 100) for dx, dz in product(range(-1, 2), range(-1, 2))):
            return False  # along stair case or door
        if not any(AREA_STRUCTURE[pos + (dx, 0, dz)] in (6, 8) for dx, dz in ((0, 1), (1, 0), (0, -1), (-1, 0))):
            return False  # not along a wall
        return True

    @staticmethod
    def __decorate_wall(wall_position):
        direction = RoomLookAround.get_direction(wall_position)
        case: int = random.choices([0, 1], weights=[3, 1])[0]
        if case == 0:
            block_state = BlockAPI.getTorch(facing=direction.name.lower())
            AREA_STRUCTURE.set(wall_position, block_state, RoomFurnisher.PRIORITY)
        elif case == 1:
            decoration = random.choice(wall_decorations)
            entity = get_item_frame_entity(decoration, facing=direction, invisible=True)
            entity.move_to(wall_position.abs_xyz)

    @staticmethod
    def __decorate_surface(wall_position):
        case: int = random.choices([0, 1], weights=[3, 1])[0]

        block_state: str = BlockAPI.blocks.Air
        if case == 0:
            plant = random.choice(["dandelion", "poppy", "blue_orchid", "allium", "azure_bluet", "red_tulip", "orange_tulip", "white_tulip", "pink_tulip", "oxeye_daisy", "cornflower", "lily_of_the_valley", "wither_rose"])
            block_state = f'potted_{plant.split(":")[-1]}'
        elif case == 1:
            block_state = BlockAPI.blocks.Lantern
        FurniturePlacer(block_state).place(wall_position)

