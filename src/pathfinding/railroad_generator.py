import logging
from typing import List, Set

import numpy as np

from .railway import TrainStation
from generation import Generator
from generation.structure import AREA_STRUCTURE
from utils import TransformBox, Position, BlockAPI as b, clear_tree_at, Point, mean, manhattan, Direction
from utils.algorithms.graphs import connected_components, GridGraph
from utils.block_utils import build_block_state, water_blocks


class RailRoadGenerator(Generator):
    NEIGHBOUR_GRAPH = GridGraph(False)
    BALLAST_PALETTE = [b.blocks.Granite] * 3 + [b.blocks.CoarseDirt] * 2 + [b.blocks.DeadFireCoralBlock, b.blocks.JunglePlanks, b.blocks.Dirt]
    PRIORITY = 20

    def __init__(self, tracks_array, stations, **kwargs):
        super().__init__(TransformBox(), **kwargs)
        self.__is_path: np.ndarray[bool] = tracks_array
        self.__track_positions = {Position(*xz) for xz in zip(*np.where(tracks_array))}
        self.__stations: List[TrainStation] = stations

        padded_path_array = np.pad(self.__is_path, 1).astype(int) * 2  # 2 -> spacing between rails
        padded_rail_array = np.zeros(padded_path_array.shape)
        for x, z in zip(*np.where(padded_path_array == 2)):
            padded_rail_array[(x-1): (x+2), (z-1): (z+2)] = 1  # 1 -> rails
        padded_rail_array[padded_path_array == 2] = 0  # un-rail path blocks

        self.__rail_array = padded_rail_array[1:-1, 1:-1].astype(bool)  # un-pad array
        self.__rail_positions: Set[Position] = {Position(*xz) for xz in zip(*np.where(self.__rail_array))}
        self.__accelerators: Set[Position] = set()

    def generate(self, level, height_map=None, palette=None):
        super().generate(level, height_map, palette)
        height_map = self.__compute_height_map(height_map)

        for rail_pos in self.__rail_positions:
            clear_tree_at(level, Point(rail_pos.abs_x, rail_pos.abs_z))
            rail_texture = self.get_rail_blockstate(rail_pos, height_map)
            rail_pos = rail_pos + Position(0, 0, height_map[rail_pos.x, rail_pos.z] + 1)
            x, y, z = rail_pos.abs_x, rail_pos.y, rail_pos.abs_z
            logging.debug(x, y, z, rail_texture)
            ballast_box, air_box = TransformBox((x - 1, y - 1, z - 1), (3, 4, 3)).split(dy=1)  # type: TransformBox, TransformBox
            protected_box = air_box.expand(1, 1, 1)
            # AREA_STRUCTURE.fill(protected_box, b.blocks.Stone, self.PRIORITY, replace=list(water_blocks))  # VERY SLOW
            AREA_STRUCTURE.fill(ballast_box, self.BALLAST_PALETTE, self.PRIORITY + 1)
            AREA_STRUCTURE.fill(air_box, b.blocks.Air, self.PRIORITY + 2)
            below_rail_block = b.blocks.RedstoneBlock if 'powered' in rail_texture else self.BALLAST_PALETTE
            AREA_STRUCTURE.set(rail_pos - Position(0, 0, 1), below_rail_block, self.PRIORITY + 3)
        for rail_pos in self.__rail_positions:
            rail_texture = self.get_rail_blockstate(rail_pos, height_map)
            rail_pos = rail_pos.withCoords(y=height_map[rail_pos.x, rail_pos.z] + 1)
            AREA_STRUCTURE.set(rail_pos, rail_texture, self.PRIORITY + 3)

    def get_rail_blockstate(self, rail_pos: Position, height_map: np.ndarray):
        neighbours = list(self.NEIGHBOUR_GRAPH.getNeighbours(rail_pos).intersection(self.__rail_positions))
        if not (1 <= len(neighbours) <= 2):
            logging.error(f"{len(neighbours)} rail neighbours found at {rail_pos}")
            return b.blocks.Rail

        # Straight rail
        if len(neighbours) == 1 or 0 in (neighbours[0] - neighbours[1]).xz:
            try:
                # if one neighbour is higher, builds sloped powered rail
                ascending_neighbour = next(_ for _ in neighbours if height_map[rail_pos.xz] < height_map[_.xz])
                ascending_direction: Direction = Direction.of(*(ascending_neighbour - rail_pos).xyz)
                return build_block_state(b.blocks.PoweredRail, shape=f"ascending_{ascending_direction.name.lower()}")
            except StopIteration:
                rail_direction: Direction = Direction.of(*(neighbours.pop() - rail_pos).xyz)
                shape = 'north_south' if rail_direction in [Direction.North, Direction.South] else 'east_west'
                block = b.blocks.PoweredRail if rail_pos in self.__accelerators else b.blocks.Rail
                return build_block_state(block, shape=shape)
        else:
            neighbour_dirs = [Direction.of(*(neighbour - rail_pos).xyz).name.lower() for neighbour in neighbours]
            if neighbour_dirs[1] in ['north', 'south']:
                neighbour_dirs = reversed(neighbour_dirs)  # north/south first, east/west next
            return build_block_state(b.blocks.Rail, shape='_'.join(neighbour_dirs))

    def __compute_height_map(self, height_map: np.ndarray):
        railways: List[Set[Position]]  # rail points, grouped by connected line
        section_list: List[List[Position]]  # alternates curves and straight lines (starts with a curve)
        height_list: List[float]  # list of heights for each section of section_list
        connector_heights = {}
        for station in self.__stations:
            connector_heights.update({conn: station.position.y for conn in station.connectors})
        fixed_height_sections = set()

        def is_in_straight_section(point):
            try:
                _nb1, _nb2 = self.NEIGHBOUR_GRAPH.getNeighbours(point).intersection(self.__rail_positions)
                return 0 in (_nb1 - _nb2).xz
            except ValueError:
                return False

        def split_railway_in_lines_and_curves(railway: Set[Position]):
            sections = [[railway.pop()]]

            currently_in_straight_section = is_in_straight_section(sections[-1][-1])

            while railway:
                # iteratively gets an unexplored neighbour to the last explored point and append it to the structure
                neighbour = self.NEIGHBOUR_GRAPH.getNeighbours(sections[-1][-1]).intersection(railway).pop()
                railway.remove(neighbour)

                if currently_in_straight_section ^ is_in_straight_section(neighbour):  # change in section type
                    sections.append([neighbour])
                    currently_in_straight_section = not currently_in_straight_section
                else:
                    sections[-1].append(neighbour)

            if len(sections) % 2 == 1 and len(sections) > 1:
                first_section = sections.pop(0)
                sections[-1].extend(first_section)
            if is_in_straight_section(sections[0][0]):
                sections = sections[1:] + [sections[0]]  # always start with a curve
            return sections

        def valid_config():
            section_count = len(section_list)
            for curve_index in range(0, section_count, 2):
                curve_height = height_list[curve_index]
                next_straight_length = len(section_list[curve_index + 1])
                next_curve_height = height_list[(curve_index + 2) % section_count]
                elevation = abs(round(curve_height) - round(next_curve_height))
                if elevation > next_straight_length:
                    return False
            return True

        def update_heights() -> None:
            section_count = len(section_list)
            for curve_index in range(0, section_count, 2):
                if curve_index in fixed_height_sections:
                    continue
                curve_height = height_list[curve_index]

                section_length = sum(len(section_list[i % section_count]) for i in range(curve_index - 1, curve_index + 2))

                for delta_index in (-2, 2):
                    neighbour_curve_index = (curve_index + delta_index) % section_count
                    neighbour_curve_height = height_list[neighbour_curve_index]
                    elevation = (neighbour_curve_height - curve_height)  # if neighbour is higher, elevation is >= 0
                    height_list[curve_index] += elevation / section_length  # if elevation is >= 0, curve rises

        rail_height_map = np.full(height_map.shape, 256)
        railways = connected_components(self.__rail_positions)
        for section_list in map(split_railway_in_lines_and_curves, railways):
            height_list = [mean(height_map[p.xz] for p in sec) for sec in section_list]

            for sec_index, sec in enumerate(section_list):
                if min(min(manhattan(p, c.pos) for p in sec) for c in connector_heights) <= 1 and not is_in_straight_section(sec[0]):
                    conn = next(c for c in connector_heights if min(manhattan(p, c.pos) for p in sec) == 1)
                    sec_height = connector_heights[conn]
                    # for neigh_sec_index in range(sec_index - 1, sec_index + 2):
                    for neigh_sec_index in [sec_index - sec_index % 2]:
                        neigh_sec_index %= len(section_list)
                        height_list[neigh_sec_index] = sec_height
                        fixed_height_sections.add(neigh_sec_index)

            while not valid_config():
                update_heights()

            height_list = [round(h) for h in height_list]

            for index, section in enumerate(section_list):
                if index % 2 == 1:
                    # straight section
                    prev_height = height_list[index - 1]
                    next_height = height_list[(index + 1) % len(height_list)]
                    if next_height > prev_height:
                        next_height -= 1
                    elif next_height < prev_height:
                        prev_height -= 1
                    heights = map(round, np.linspace(prev_height, next_height, len(section)))
                else:
                    heights = [height_list[index]] * len(section)

                if manhattan(section[-1], section_list[index - 1][-1]) < manhattan(section[0], section_list[index - 1][-1]):
                    section = list(reversed(section))
                for section_point, point_height in zip(section, heights):
                    rail_height_map[section_point.xz] = point_height
                self.__accelerators.add(section[0])

        return rail_height_map
