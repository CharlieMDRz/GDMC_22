import itertools
import logging
from math import floor
import random
from typing import List, Tuple

import numpy as np
from gdpc import direct_interface
from numpy import percentile
from numpy.random import choice as npChoice
from scipy import ndimage

from generation.building_palette import HousePalette
from generation.structure import AREA_STRUCTURE
from parameters import ACTIVATED_WINDMILLS
from utils import *
from utils.block_utils import build_block_state
from utils.nbt_structures import StructureNBT

SURFACE_PER_ANIMAL = 16
activated_windmills = ACTIVATED_WINDMILLS


class Generator:
    _box: BoundingBox = None

    def __init__(self, box: BoundingBox, **kwargs):
        self._box = box
        self._entry_point: Position = kwargs.get("entry_point", Position(0, 0))
        self.children = []  # type: List[Generator]
        self._sub_generator_function = None

    def _clear_trees(self, level):
        for x, z in itertools.product(range(self._box.minx, self._box.maxx), range(self._box.minz, self._box.maxz)):
            clear_tree_at(level, Point(x, z))

    def surface_pos(self, height_map):
        for x, z in itertools.product(range(self.width), range(self.length)):
            yield x + self.origin.x, height_map[x, z], z + self.origin.z

    def choose_sub_generator(self, parcels):
        pass

    def generate(self, level, height_map=None, palette=None):
        """
        Generates this building on the level
        Parameters
        ----------
        level MC world - TerrainMaps instance
        height_map optional height_map

        Returns nothing
        -------

        """
        for sub_generator in self.children:
            sub_generator.generate(level, height_map, palette)

    def is_corner(self, pos):
        return Generator.is_lateral(self, pos.x) and Generator.is_lateral(self, None, pos.z)

    def is_lateral(self, x=None, z=None):
        assert x is not None or z is not None
        z_lateral = (z == self._box.minz or z == self._box.maxz - 1)
        x_lateral = (x == self._box.minx or x == self._box.maxx - 1)
        if x is None:
            return z_lateral
        if z is None:
            return x_lateral
        else:
            return x_lateral or z_lateral

    @property
    def width(self):
        return self._box.width

    @property
    def length(self):
        return self._box.length

    @property
    def height(self):
        return self._box.height

    @property
    def origin(self):
        return self._box.origin

    @property
    def size(self):
        return self._box.size

    @property
    def surface(self):
        return self._box.surface

    @property
    def mean(self) -> Position:
        return Position(self._box.minx + self.width // 2, self._box.minz + self.length // 2, absolute_coords=True)

    def translate(self, dx=0, dy=0, dz=0):
        self._box.translate(dx, dy, dz, True)
        for gen in self.children:
            gen.translate(dx, dy, dz)

    @property
    def entry_direction(self):
        entry_dir_vec: Point = self._entry_point - self.mean
        try:
            return Direction.of(dx=entry_dir_vec.x, dz=entry_dir_vec.z)
        except AssertionError:
            return list(Direction.cardinal_directions())[0]

    def absolute_coords(self, x, z):
        return x + self._box.minx, z + self._box.minz


class MaskedGenerator(Generator):
    def __init__(self, box, **kwargs):
        Generator.__init__(self, box, **kwargs)
        self._mask = kwargs.get("mask", np.full((box.width, box.length), True))

    def _terraform(self, level, height_map):
        # type: (TerrainMaps, array) -> np.ndarray
        mean_y = int(round(height_map.mean()))
        terraform_map = np.zeros(height_map.shape)
        for x, y, z in self.surface_pos(height_map):
            if y > mean_y:
                vbox = BoundingBox((x, mean_y + 1, z), (1, y - mean_y + 1, 1))
                AREA_STRUCTURE.fill(vbox, alpha.Air)
            elif y < mean_y:
                vbox = BoundingBox((x, y + 1, z), (1, mean_y - y, 1))
                material = alpha.StoneBricks if self.is_lateral(x, z) else alpha.Dirt
                AREA_STRUCTURE.fill(vbox, material, 2)
        terraform_map[:] = mean_y
        return terraform_map

    def is_masked(self, px, z=None, absolute_coords=False):
        if z is None:
            return self.is_masked(px.x, px.z, absolute_coords)
        else:
            if absolute_coords:
                return self.is_masked(px - self.origin.x, z - self.origin.z, False)
            else:
                if 0 <= px < self.width and 0 <= z < self.length:
                    return self._mask[px, z]
                return False

    def surface_pos(self, height_map):
        for x, y, z in Generator.surface_pos(self, height_map):
            if self.is_masked(x, z, True):
                yield x, y, z

    def is_corner(self, pos):
        if Generator.is_corner(self, pos):
            return self.is_masked(pos, absolute_coords=True)  # box corner in mask
        elif not self.is_masked(pos, absolute_coords=True):
            return False
        else:
            # internal point to the box, could still be a corner
            try:
                x_lateral = not (self.is_masked(pos + Direction.East.value, absolute_coords=True)
                                 and self.is_masked(pos + Direction.West.value, absolute_coords=True)
                                 )
            except IndexError:
                x_lateral = False

            try:
                z_lateral = not (self.is_masked(pos + Direction.South.value, absolute_coords=True)
                                 and self.is_masked(pos + Direction.North.value, absolute_coords=True)
                                 )
            except IndexError:
                z_lateral = False
            return x_lateral and z_lateral

    def is_lateral(self, x=None, z=None):
        if not (0 <= x < self.width and 0 <= z < self.length):
            return False
        elif Generator.is_lateral(self, x, z):
            return self.is_masked(x, z, True)
        elif not self.is_masked(x, z, True):
            return False
        else:
            assert x is not None and z is not None
            pos = Point(x, z)
            return any(not self.is_masked(pos + dir.asPoint, absolute_coords=True) for dir in Direction.cardinal_directions())

    def _clear_trees(self, level):
        x0, z0 = self.origin.x, self.origin.z
        for x, z in itertools.product(range(self._box.minx, self._box.maxx), range(self._box.minz, self._box.maxz)):
            if self._mask[x - x0, z - z0]:
                clear_tree_at(level, Point(x, z))

    def add_mask(self, height_mask):
        assert self._mask.shape == height_mask.shape
        self._mask = self._mask & height_mask

    def refine_mask(self):
        """
        Removes narrow positions from the mask
        """
        def surrounding(x, z):
            return filter(lambda xz: xz != (x, z) and self.is_masked(xz[0], xz[1], True), itertools.product(range(x-1, x+2), range(z-1, z+2)))

        def is_corner(xz):
            return self.is_corner(Point(*xz))

        def in_inner_mask(xz):
            return self.is_masked(*xz, True) and not self.is_lateral(*xz)

        mask_corners = {(x, z) for (x, y, z) in self.surface_pos(np.zeros((self.width, self.length))) if self.is_corner(Point(x, z))}
        while mask_corners:
            corner = mask_corners.pop()
            if all(not in_inner_mask(xz) for xz in surrounding(*corner)):
                self._mask[corner[0] - self.origin[0], corner[1] - self.origin[2]] = False
                mask_corners.update(filter(is_corner, surrounding(*corner)))


class CropGenerator(MaskedGenerator):

    animal_distribution = {"cow": 3, "pig": 2, "chicken": 3, "sheep": 2, "donkey": 1, "horse": 1, "rabbit": 1}

    def _pick_animal(self):
        samples = list(self.animal_distribution.keys())
        probs = np.array(list(self.animal_distribution.values()))
        probs = probs / sum(probs)
        return npChoice(samples, p=probs)

    def choose_sub_generator(self, parcels):
        # type: (List[Parcel]) -> None
        self._sub_generator_function = self._gen_animal_farm
        if any(_.building_type.name == 'windmill' for _ in parcels) and bernouilli():
            d = min(euclidean(self.mean, _.absolute_mean) for _ in parcels if _.building_type.name == 'windmill')
            if d <= 16:
                self._sub_generator_function = self._gen_harvested_crop
            elif d <= 28:
                self._sub_generator_function = self._gen_crop_v1
            else:
                self._sub_generator_function = self._gen_animal_farm
        elif bernouilli(.7):
            self._sub_generator_function = self._gen_crop_v1
        else:
            self._sub_generator_function = self._gen_animal_farm

    def generate(self, level, height_map=None, palette=None):
        if self._mask.sum() < 16:
            logging.info(f"Parcel ({self.width}, {self.length}) at {self.mean} too small to generate a crop")
            return
        self._clear_trees(level)
        if self._sub_generator_function == self._gen_animal_farm:
            self._gen_animal_farm(height_map, palette, entities=level.entities)
        else:
            self._sub_generator_function(height_map, palette)

    def _gen_animal_farm(self, height_map, palette, animal=None, entities=None):
        # type: (TerrainMaps, array, HousePalette, str) -> None
        from terrain.entity_manager import EntityManager
        entities: EntityManager
        self.refine_mask()
        if not animal:
            if entities:
                from terrain.entity_manager import get_most_populated_animal
                animal = get_most_populated_animal(entities)
            if not (entities and animal):
                animal = self._pick_animal()
        print(f"Animal farm of type {animal}")
        fence_box = TransformBox(self.origin, (self.width, 1, self.length)).expand(-1, 0, -1)
        fence_block = BlockAPI.getFence(palette['door'])
        gate_pos, gate_dist = None, 0

        height_map_max = ndimage.maximum_filter(height_map, size=3)

        # place fences
        for ax, y, az in self.surface_pos(height_map):
            x, z = ax - self.origin.x, az - self.origin.z
            if not self.is_masked(ax, az, True):
                continue
            if self.is_lateral(ax, az):
                box = BoundingBox((ax, y + 1, az), (1, height_map_max[x, z] - y + 1, 1))
                AREA_STRUCTURE.fill(box, fence_block, 4)
                new_gate_pos = Point(ax, az, y+1)
                new_gate_dist = euclidean(Position(ax, az, y+1, True), self._entry_point)
                if (gate_pos is None or new_gate_dist < gate_dist) and not self.is_corner(new_gate_pos) and box.height == 1:
                    gate_pos, gate_dist = new_gate_pos, new_gate_dist

        if gate_pos:
            for direction in Direction.cardinal_directions(False):
                if not self.is_masked(gate_pos + direction.value, absolute_coords=True):
                    door_dir = direction
                    gate_block = BlockAPI.getFence(palette['door'], facing=str(door_dir).lower())
                    AREA_STRUCTURE.set(gate_pos, gate_block, 5)
                    for dir in (door_dir.rotate(), -door_dir.rotate()):  # type: Direction
                        torch_pos: Point = (gate_pos + dir.value)
                        y = height_map_max[torch_pos.x - self.origin.x, torch_pos.z - self.origin.z]
                        place_torch(torch_pos.x, y + 2, torch_pos.z)
                    break

        # place animals
        animal_count = sum(self._mask.flat) // SURFACE_PER_ANIMAL
        while animal_count > 0:
            ax = random.randint(fence_box.minx, fence_box.maxx - 1)
            az = random.randint(fence_box.minz, fence_box.maxz - 1)
            y = height_map[ax - self.origin.x, az - self.origin.z] + 1
            if self.is_masked(ax, az, True) and not self.is_lateral(ax, az):
                if entities:
                    entity = entities.get_entity(animal, position=Position(ax, az, absolute_coords=True))
                else:
                    entity = Entity(animal)
                if bernouilli(.3):
                    entity["Age"] = random.randint(-25000, -5000)
                entity.move_to((ax, y, az))
                animal_count -= 1
            else:
                animal_count -= .1

    def _gen_crop_v1(self, height=None, palette=None):
        min_height = int(percentile(height.flatten(), 20))
        max_height = int(percentile(height.flatten(), 80))
        self._mask &= (height >= min_height) & (height <= max_height)
        # block states
        b = alpha
        crop_type, max_age = random.choice([(b.Carrots, 7), (b.Beetroots, 3), (b.Potatoes, 7), (b.Wheat, 7)])
        crop_age = random.randint(max_age // 4, 1 + max_age * 3 // 4)

        # Place crops
        for x, y, z in self.surface_pos(height):
            # farmland
            AREA_STRUCTURE.set(Point(x, z, y), f"farmland[moisture=7]", 3)
            # crop
            crop_age = crop_age + random.random() - .5
            int_crop_age = pos_bound(int(round(crop_age)), max_age)
            crop_block = f"{crop_type}[age={int_crop_age}]"
            AREA_STRUCTURE.set(Point(x, z, y+1), crop_block, 2)

        self._irrigate_field(height, 4)

    def _gen_harvested_crop(self, height_map, palette=None):
        mx, mz = random.randint(0, 1), random.randint(0, 2)
        for x, y, z in self.surface_pos(height_map):
            if (x % 2 == mx and (z + x // 2) % 3 == mz) and bernouilli():
                AREA_STRUCTURE.set(Point(x, z, y), alpha.Dirt)  # dirt under hay bales
                b = alpha.HayBlock
                y += 1
            else:
                b = alpha.Farmland
            AREA_STRUCTURE.set(Point(x, z, y), b)
        place_water_source(self.mean.x, y, self.mean.z)

    def _irrigate_field(self, height_map, spread):
        """
        Irrigate the field by sampling dry positions until there is none
        :param height_map: matrix of heights of the field
        :param spread: max distance from water for crops to grow
        """
        x0, y0, z0 = self.origin
        water_sources = set()
        irrigated: np.ndarray = np.copy(self._mask).astype(int)
        irrigated[:] = 0
        irrigated[1:-1, 1:-1] = 1  # make border points as irrigated so water won't spawn on them
        irrigated = np.minimum(irrigated, self._mask[:].astype(int))

        def sample_dry_pos() -> Tuple[int, int, int]:
            dry_pos = np.where(irrigated == 1)
            index = np.random.randint(len(dry_pos[0]))
            x, z = dry_pos[0][index], dry_pos[1][index]
            y = height_map[x, z]
            return x, y, z

        while irrigated.sum():
            rx, y_water, rz = sample_dry_pos()
            ax, az = rx + x0, rz + z0
            water_sources.add((ax, az))

            p = Point(ax, az, y_water)
            for dir in Direction.cardinal_directions(False):  # type: Direction
                neighbour: Point = p + dir.value
                y_crop = height_map[neighbour.x - x0, neighbour.z - z0]
                if y_crop < y_water:  # Prevents overflowing to adjacent positions
                    state = f"spruce_trapdoor[facing={dir.name.lower()}, half=bottom, open=true]"
                    AREA_STRUCTURE.set(neighbour, state, 5)
            AREA_STRUCTURE.set(p, alpha.Water, 5)

            mask = (height_map == y_water)[max(0, rx-spread):(rx+spread+1), max(0, rz-spread):(rz+spread+1)]
            irrigated[max(0, rx-spread):(rx+spread+1), max(0, rz-spread):(rz+spread+1)][mask] = 0

    def is_lateral(self, x=None, z=None):
        p = Point(x, z)
        if Generator.is_lateral(self, x, z):
            return True
        if not self.is_masked(p, absolute_coords=True):
            return False
        x0, z0 = x - self.origin.x, z - self.origin.z
        for x1, z1 in itertools.product(sym_range(x0, 1, self.width), sym_range(z0, 1, self.length)):
            if not self.is_masked(x1, z1):
                return True
        return False


class CardinalGenerator(Generator):
    """
    Generator linked to its direct neighbors in each direction (N, E, S, W, top, bottom)
    """

    def __init__(self, box):
        Generator.__init__(self, box)
        self._neighbors = dict()

    def __getitem__(self, item):
        # type: (object) -> CardinalGenerator
        if isinstance(item, Direction) and item in self._neighbors:
            # find a sub generator by its direction
            return self._neighbors[item]
        elif isinstance(item, TransformBox):
            # find a sub generator by its box
            for sub_item in self.children:
                if sub_item.origin == item.origin and item.size == sub_item.size:
                    return sub_item
        return None

    def __setitem__(self, direction, neighbour):
        # type: (Direction, CardinalGenerator) -> None
        """
        Marks neighbouring relationship between two generators
        """
        if isinstance(direction, Direction) and isinstance(neighbour, CardinalGenerator):
            self._neighbors[direction] = neighbour
            # Upper floor neighbours are descendants of lower floors. If this method is called with rooms from the upper
            # floors, then <neighbour> already has a parent: the room underneath -> no new parenting link
            if neighbour[Direction.Bottom] is None:
                self.children.insert(0, neighbour)
            neighbour._neighbors[-direction] = self
            if direction == Direction.Top:
                for direction2 in Direction.cardinal_directions(False):
                    if self[direction2] is not None and self[direction2][Direction.Top] is not None:
                        neighbour[direction2] = self[direction2][Direction.Top]
        else:
            raise TypeError


class DoorGenerator(Generator):
    def __init__(self, box, direction, material='Oak'):
        # type: (TransformBox, Direction, str) -> DoorGenerator
        Generator.__init__(self, box)
        assert 1 <= box.surface <= 2
        self._direction = direction
        self._material = material

    def generate(self, level, height_map=None, palette=None):
        for x, y, z in self._box.positions:
            state = self._resource(x, y, z, palette)
            AREA_STRUCTURE.set(Point(x, z, y), state, 100)
        AREA_STRUCTURE.fill(self._box.translate(self._direction).split(dy=2)[0], alpha.Air, 100)

    def _resource(self, x, y, z, palette):
        if self._box.miny <= y <= self._box.miny + 1:
            half = "lower" if y == self._box.miny else "upper"
            if self._box.surface == 1:
                hinge = 'left'
            else:
                mean_x = self._box.minx + 0.5 * self._box.width
                mean_z = self._box.minz + 0.5 * self._box.length
                norm_dir = self._direction.rotate()
                left_x = int(floor(mean_x + 0.5 * norm_dir.x))  # floored because int(negative float) rounds up
                left_z = int(floor(mean_z + 0.5 * norm_dir.z))
                hinge = 'left' if (x == left_x and z == left_z) else 'right'
            block_name = BlockAPI.getDoor(palette['door'], half=half, hinge=hinge, facing=(-self._direction).name.lower())
        else:
            block_name = palette['wall']
        return block_name


class WindmillGenerator(Generator):
    def generate(self, level, height_map=None, palette=None):

        # Compute windmill position
        box = self._box
        if box.width < 7 or box.length < 7:
            return
        x, z = box.minx + box.width // 2, box.minz + box.length // 2
        y = height_map[box.width//2, box.length//2]


        # Build floor
        ground_box = TransformBox((x-2, y, z-2), (5, 1, 5))
        AREA_STRUCTURE.fill(ground_box, alpha.CoarseDirt)

        # Build windmill frames
        box = TransformBox((x-5, y-31, z-4), (11, 11, 8))
        AREA_STRUCTURE.fill(box.expand(1), alpha.Bedrock, 999)  # protective shell around windmill frames
        mech_nbt = StructureNBT('gdmc_windmill_mech.nbt')
        mech_nbt.build(*box.origin)

        # Build windmill
        box.translate(dy=31, inplace=True)
        windmill_nbt = StructureNBT('gdmc_windmill.nbt')
        windmill_nbt.build(*box.origin)
        # print(runCommand(f'setblock {x} {y+4} {z-1} minecraft:redstone_wall_torch[facing=north, lit=true]'))
        global activated_windmills
        if activated_windmills:
            direct_interface.runCommand(f'setblock {x} {y+4} {z-1} minecraft:redstone_wall_torch[facing=north, lit=true]')
            activated_windmills -= 1


def place_street_lamp(x, y, z, material, h=0):
    h = max(1, 3+h)
    AREA_STRUCTURE.fill(BoundingBox((x, y + 1, z), (1, h, 1)), BlockAPI.getFence(material), 5)
    AREA_STRUCTURE.set(Point(x, z, y + h + 1), alpha.RedstoneLamp, 5)
    AREA_STRUCTURE.set(Point(x, z, y + h + 2), build_block_state(alpha.DaylightDetector, inverted='true'), 5)


def place_torch_post(x, y, z, block=None):
    if block is None:
        block = random.choice([alpha.OakFence, alpha.CobblestoneWall, alpha.MossyCobblestoneWall, alpha.SpruceFence])
    AREA_STRUCTURE.set(Point(x, z, y+1), block, 5)
    place_torch(x, y+2, z)


def place_water_source(x, y, z, protected_points=None):
    p = Point(x, z, y)
    for dir in Direction.cardinal_directions(False):  # type: Direction
        dpos = p + dir.value
        if direct_interface.getBlock(*dpos.xyz).split(':')[-1] not in ground_blocks:
            state = f"spruce_trapdoor[facing={dir.name.lower()}, half=bottom, open=true]"
            AREA_STRUCTURE.set(dpos, state, 5)
            if protected_points:
                protected_points.add((dpos.x, dpos.z))
    AREA_STRUCTURE.set(p, alpha.Water, 5)


def sign_rotation_for_direction(p: Point):
    from math import acos, pi
    rad = acos(p.x / p.norm)
    if p.z < 0:
        rad = 2 * pi - rad
    ang = rad / pi * 8
    return int(round((ang + 4))) % 16


def place_sign(position: Point, material: str, direction: Point, **kwargs):
    """
    Place a sign at a given position
    :param position:
    :param material:
    :param direction: direction faced by the sign
    :param kwargs:
    :keyword Text1: (str) First line of the sign. Goes on to Text4
    :keyword Color: (str) Color of the text
    :return: nothing
    """
    if 'wall' in material:
        state = material + f"facing={Direction.of(*position.xyz).name.lower()}" + "{"
    else:
        state = material + f"[rotation={sign_rotation_for_direction(direction)}]" + "{"
    state += f'Color: "{kwargs.get("Color", "black")}", '
    state += f'x: {position.x}, y: {position.y}, z: {position.z}, id: "minecraft:sign", '
    state += 'Text1: \'{"text":"' + kwargs.get("Text1", "") + '"}\', '
    state += 'Text2: \'{"text":"' + kwargs.get("Text2", "") + '"}\', '
    state += 'Text3: \'{"text":"' + kwargs.get("Text3", "") + '"}\', '
    state += 'Text4: \'{"text":"' + kwargs.get("Text4", "") + '"}\''
    state += "}"
    command = f"setblock {position.x} {position.y} {position.z} {state}"
    direct_interface.runCommand(command)


if __name__ == '__main__':
    gen = CropGenerator(TransformBox((0, 0, 0), (1, 1, 1)))
    print(gen.animal_distribution)
    print(gen._pick_animal())
    for dir in Direction.cardinal_directions(False):
        print(dir.name, sign_rotation_for_direction(dir.value))
