import time
from typing import Union

from gdpc import worldLoader

from generation.generators import *
from generation.structure import AREA_STRUCTURE
from utils import bernouilli, Direction


class ProcHouseGenerator(MaskedGenerator):

    def generate(self, level, height_map=None, palette=None):
        t0 = time.time()
        n_iter = 5000
        main_building: _RoomSymbol = ProcHouseGeneratorBuilder(self.origin, self._mask).build(self._box, n_iter)
        print(f"Computed {n_iter} foot prints in {(time.time() - t0):0.3}s !")

        if main_building is None:
            print("Parcel ({}, {}) at {} too small to generate a house".format(self.width, self.length, self.mean))
            return
        else:
            self.children.append(main_building)
            self._clear_trees(level)
            Generator.generate(self, level.level, height_map, palette)
            main_room: _RoomSymbol = self.children[0]
            main_room.generate_door(self.entry_direction, self._entry_point.abs_x, self._entry_point.abs_z, level, palette)
            self._generate_stairs(level.level, palette)

    def _clear_trees(self, level):
        for gen in filter(lambda g: isinstance(g, _RoomSymbol), self.children):
            gen._clear_trees(level)

    def _center_building(self):
        dx = int(round((self._box.width - self._layout_width) / 2))
        dz = int(round((self._box.length - self._layout_length) / 2))
        self.children[0].translate(dx, 0, dz)

    def _generate_stairs(self, level, palette):
        main_room: _RoomSymbol = self.children[0]
        door_wall: Direction = self.entry_direction
        stair_wall: Direction or None = None

        # check room size for long walls
        possible_stair_dir = set()
        if main_room.length >= 7 and main_room.width > 5: possible_stair_dir.update((Direction.West, Direction.East))
        if main_room.width >= 7 and main_room.length > 5: possible_stair_dir.update((Direction.North, Direction.South))

        # remove unsuitable walls
        possible_stair_dir.difference_update({door_wall})
        possible_stair_dir.difference_update(main_room._neighbors.keys())

        if possible_stair_dir:
            stair_wall = random.choice(list(possible_stair_dir))  # if there remains suitable ones, pick one

        revved = False
        if stair_wall:
            while isinstance(main_room[Direction.Top], _RoomSymbol):
                main_room.generate_stairs(stair_wall, palette, reversed=revved)
                revved = not revved
                main_room = main_room[Direction.Top]
        elif isinstance(main_room[Direction.Top], _RoomSymbol):
            main_room.generate_ladder()


class _RoomSymbol(CardinalGenerator):

    def __init__(self, box, has_base=False):
        CardinalGenerator.__init__(self, box)
        self._has_base = has_base

    def generate(self, level, height_map=None, palette=None):
        # log.debug("Generating Room at", self._get_box().origin, self._get_box().size)
        if self._has_base:
            h = 1 if height_map is None else max(1, self.origin.y - height_map.min())
            self.children.append(_BaseSymbol(TransformBox(self.origin - (0, h, 0), (self.width, h, self.length))))
        AREA_STRUCTURE.fill(self._get_box(), BlockAPI.blocks.Air, 5)
        self._generate_pillars(level, palette)
        self._create_walls(level, palette)
        self.__place_torch(level)

        prob = self._box.height / 4 - 1  # probability to build an upper floor
        upper_box = self._get_upper_box()

        if bernouilli(prob):
            ceiling_box = upper_box.translate(dy=-1).split(dy=1)[0]
            AREA_STRUCTURE.fill(ceiling_box.expand(-1, 0, -1), palette['floor'], 11)
            upper_room = _RoomSymbol(upper_box)
        else:
            upper_room = _RoofSymbol(upper_box, roof_type=palette['roofType'])
        self[Direction.Top] = upper_room
        # build upper Symbol
        Generator.generate(self, level, height_map, palette)

    def _get_box(self):
        return TransformBox(self._box.origin, (self._box.width, 4, self._box.length))

    def _get_upper_box(self):
        box = self._box
        new_origin = box.origin + (0, 4, 0)
        new_height = max(1, box.height - 4)
        new_size = (box.width, new_height, box.length)
        return TransformBox(new_origin, new_size)

    def _generate_pillars(self, level, palette):
        b = self._get_box()
        for col_box in [TransformBox(b.origin, (1, b.height, 1)),
                        TransformBox(b.origin + (0, 0, b.length - 1), (1, b.height, 1)),
                        TransformBox(b.origin + (b.width - 1, 0, 0), (1, b.height, 1)),
                        TransformBox(b.origin + (b.width - 1, 0, b.length - 1), (1, b.height, 1))]:
            AREA_STRUCTURE.fill(col_box, palette.get_structure_block('y'), 8)

    def _create_walls(self, level, palette):
        for direction in Direction.cardinal_directions(False):
            wall_box = self.get_wall_box(direction)
            if self[direction] is not None:
                if isinstance(self[direction], _RoofSymbol):
                    AREA_STRUCTURE.fill(wall_box, palette['wall'], 8)
                    continue
                elif wall_box.volume < self[direction].get_wall_box(-direction).volume:
                    # there is an extension room in that direction - creates an opening
                    wall_box, arch_box = wall_box.split(dy=3)
                    wider_box = wall_box.enlarge(direction)
                    AREA_STRUCTURE.fill(wider_box, palette.get_structure_block('y'), 8)
                    AREA_STRUCTURE.fill(wall_box, BlockAPI.blocks.Air, 10)
                    AREA_STRUCTURE.fill(arch_box, palette['floor'], 7)
                    continue

            # some annexes are only one block wide or long, could generate negative dimensions
            if wall_box.width * wall_box.length >= 1:
                self.children.insert(0, _WallSymbol(wall_box))

    def get_wall_box(self, direction):
        # type: (Direction) -> TransformBox
        b = self._get_box()
        if direction == Direction.North:
            return TransformBox(b.origin + (1, 0, 0), (b.width - 2, b.height, 1))
        elif direction == Direction.South:
            return TransformBox(b.origin + (1, 0, b.length - 1), (b.width - 2, b.height, 1))
        elif direction == Direction.West:
            return TransformBox(b.origin + (0, 0, 1), (1, b.height, b.length - 2))
        elif direction == Direction.East:
            return TransformBox(b.origin + (b.width - 1, 0, 1), (1, b.height, b.length - 2))
        else:
            raise ValueError("Not implemented yet, or unexpected direction {}".format(direction))

    def generate_door(self, parcel_door_dir, door_x, door_z, level, palette):
        # type: (Direction, int, int, MCLevel, HousePalette) -> None
        """
            Generate a door in self room
            Parameters
            ----------
            parcel_door_dir Direction of the door relative to the parcel
            door_x X coordinate of the door on the parcel border
            door_z Z coordinate of the door on the parcel border
        """
        local_door_dir = Direction.of(dx=door_x-self.mean.abs_x, dz=door_z-self.mean.abs_z)  # direction of the door relative to this room
        if self[local_door_dir] is not None and isinstance(self[local_door_dir], _RoomSymbol):
            try:
                # passes the door to an annex room
                self[local_door_dir].generate_door(local_door_dir, door_x, door_z, level, palette)
            except RuntimeError:
                local_door_dir = local_door_dir.rotate()
                door_wall_box = self.get_wall_box(local_door_dir)
                self[door_wall_box].generate_door(local_door_dir, door_x, door_z, level, palette)
            finally:
                return
        else:
            # passes the door to the most suited wall of the room (no annex, close to entrance & large enough)
            door_dir = local_door_dir if self.get_wall_box(local_door_dir).surface > 1 else parcel_door_dir
            door_wall_box = self.get_wall_box(door_dir)
            self[door_wall_box].generate_door(door_dir, door_x, door_z, level, palette)

    def __place_torch(self, level):
        x0, y, z0 = self.origin + (1, 0, 1)
        xM, zM = x0 + self.width - 3, z0 + self.length - 3
        if xM >= x0 and zM >= z0:
            x, z = random.randint(x0, xM), random.randint(z0, zM)
            place_torch(x, y, z)

    def generate_ladder(self):
        b = self._box
        border_pts = set(itertools.product(range(b.minx, b.maxx), range(b.minz, b.maxz)))
        border_pts.difference_update(itertools.product(range(b.minx + 1, b.maxx - 1), range(b.minz + 1, b.maxz - 1)))

        def valid_pos(x: int, y: int, z: int) -> bool:
            if (x in (self._box.minx, self._box.maxx - 1) and z in (self._box.minz, self._box.maxz - 1)):
                return False
            block = direct_interface.getBlock(x, y, z)
            return all(_ not in block for _ in ("door", "air", "glass"))

        y = self._box.miny + 1
        while border_pts:
            x, z = border_pts.pop()
            if valid_pos(x, y, z):
                lad_dir = next(filter(lambda dir: (x, y, z) in self.get_wall_box(dir), Direction.cardinal_directions(False)))
                lad_box = TransformBox((x, self._box.miny, z), (1, self._box.height, 1)).translate(-lad_dir)
                lad_str = f"ladder[facing={(-lad_dir).name.lower()}]"
                AREA_STRUCTURE.fill(lad_box, lad_str, 12)
                break

    def generate_stairs(self, direction: Direction, palette, reversed=False):
        wall_vec: Point = direction.value  # direction of the wall along which the stairs are gen'd
        stair_vec: Point = abs(direction.rotate().value)  # upwards direction of the stairs
        stair_dir: Direction = Direction.of(dx=stair_vec.x, dz=stair_vec.z)
        orig_pos = self.get_wall_box(direction).origin
        stair_vec = abs(stair_vec)
        if reversed:
            stair_dir = -stair_dir
            orig_pos = Point(orig_pos.x, orig_pos.z, orig_pos.y) - (wall_vec * 2) + (stair_vec * 3)
            stair_vec = -stair_vec
        else:
            orig_pos = Point(orig_pos.x, orig_pos.z, orig_pos.y) - wall_vec + stair_vec

        upper_material: str = BlockAPI.getStairs(palette['roofAlt'], facing=stair_dir.name.lower())
        lower_material: str = BlockAPI.getStairs(palette['roofAlt'], facing=(-stair_dir).name.lower(), half='top')
        for _ in range(4):
            stair_pos: Point = orig_pos + Point(0, 0, _) + (stair_vec * _)
            AREA_STRUCTURE.set(stair_pos, upper_material, 13)
            if _:
                AREA_STRUCTURE.set(stair_pos + Direction.Bottom.value, lower_material, 13)
            AREA_STRUCTURE.fill(BoundingBox((stair_pos.x, stair_pos.y, stair_pos.z), (1, 4 - _, 1)), "air", 12)


class _RoofSymbol(CardinalGenerator):

    def __init__(self, box, direction=None, roof_type='flat'):
        # type: (TransformBox, Direction, str) -> _RoofSymbol
        CardinalGenerator.__init__(self, box)
        self._direction = direction
        self._roof_type = roof_type

        if self._direction is None:
            # sets roof direction randomly, lower roofs are preferred
            width, length = self._box.width, self._box.length
            if width < 5:
                self._direction = Direction.East
            elif length < 5:
                self._direction = Direction.South
            else:
                prob = (1. * width ** 2) / (width ** 2 + length ** 2)
                self._direction = Direction.East if (bernouilli(prob)) else Direction.South

    def generate(self, level, height_map=None, palette=None):
        if self._roof_type == 'flat':
            box = self.flat_box
            AREA_STRUCTURE.fill(box.split(dy=1)[0], palette['roofBlock'], 2)
        elif self._roof_type == 'gable':
            if self._direction in [Direction.West, Direction.East]:
                self.__gen_gable_x(level, palette)
            elif self._direction in [Direction.North, Direction.South]:
                self.__gen_gable_z(level, palette)
            else:
                raise ValueError('Expected direction str, found {}'.format(self._direction))
            self.__gen_gable_cross(level, palette)

    @property
    def flat_box(self):
        box = self._box
        height = 1 if self._direction is None else (box.width + 1) // 2 \
            if self._direction in [Direction.North, Direction.South] else (box.length + 1) // 2
        new_size = (box.width, height, box.length)
        return TransformBox(box.origin, new_size)

    @property
    def gable_box(self):
        box = self.flat_box
        box.expand(1, 0, 1, inplace=True)
        box.expand(Direction.Bottom, inplace=True)
        for _ in range(max(box.width, box.length) // 2):
            box.expand(Direction.Top, inplace=True)
        return box

    def __gen_gable_x(self, level, palette):
        # type: (MCLevel, HousePalette) -> None
        box = self.gable_box
        for index in range(box.length // 2):
            attic_box = TransformBox((box.minx + 1, box.miny + index, box.minz + index + 1),
                                     (box.width - 2, 1, box.length - 2*(index+1)))
            north_box = TransformBox((box.minx, box.miny + index, box.maxz - index - 1), (box.width, 1, 1))
            south_box = TransformBox((box.minx, box.miny + index, box.minz + index), (box.width, 1, 1))
            AREA_STRUCTURE.fill(north_box, palette.get_roof_block('bottom', 'north'), 2)
            AREA_STRUCTURE.fill(south_box, palette.get_roof_block('bottom', 'south'), 2)
            if index != 0:
                AREA_STRUCTURE.fill(north_box.translate(dy=-1), palette.get_roof_block('top', 'south'), 2)
                AREA_STRUCTURE.fill(south_box.translate(dy=-1), palette.get_roof_block('top', 'north'), 2)
                wall1, tmp = attic_box.split(dx=1)
                _, wall2 = tmp.split(dx=-1)
                AREA_STRUCTURE.fill(wall1, palette['wallAlt'], 6)
                AREA_STRUCTURE.fill(wall2, palette['wallAlt'], 6)
            else:
                AREA_STRUCTURE.fill(attic_box, palette.get_structure_block('z'), 8)
                attic_box.expand(-1, 0, -1, inplace=True)
                AREA_STRUCTURE.fill(attic_box, 'air', 9)
        # build roof ridge
        if box.length % 2 == 1:
            index = box.length // 2
            ridge_box = TransformBox((box.minx, box.miny + index, box.minz + index), (box.width, 1, 1))
            AREA_STRUCTURE.fill(ridge_box, palette.get_roof_block('bottom'), 2)
            AREA_STRUCTURE.fill(ridge_box.translate(dy=-1), palette.get_structure_block('x'), 4)

    def __gen_gable_z(self, level, palette):
        box = self.gable_box
        for index in range(box.width // 2):
            attic_box = TransformBox((box.minx + index + 1, box.miny + index, box.minz + 1),
                                     (box.width - 2*(index+1), 1, box.length - 2))
            west_box = TransformBox((box.maxx - index - 1, box.miny + index, box.minz), (1, 1, box.length))
            east_box = TransformBox((box.minx + index, box.miny + index, box.minz), (1, 1, box.length))
            AREA_STRUCTURE.fill(west_box, palette.get_roof_block('bottom', 'west'), 2)
            AREA_STRUCTURE.fill(east_box, palette.get_roof_block('bottom', 'east'), 2)
            if index != 0:
                AREA_STRUCTURE.fill(west_box.translate(dy=-1), palette.get_roof_block('top', 'east'), 2)
                AREA_STRUCTURE.fill(east_box.translate(dy=-1), palette.get_roof_block('top', 'west'), 2)
                wall1, tmp = attic_box.split(dz=1)
                _, wall2 = tmp.split(dz=-1)
                AREA_STRUCTURE.fill(wall1, palette['wallAlt'], 6)
                AREA_STRUCTURE.fill(wall2, palette['wallAlt'], 6)
            else:
                AREA_STRUCTURE.fill(attic_box, palette.get_structure_block('x'), 8)
                attic_box.expand(-1, 0, -1, inplace=True)
                AREA_STRUCTURE.fill(attic_box, 'air', 9)
        # build roof ridge
        if box.width % 2 == 1:
            index = box.width // 2
            ridge_box = TransformBox((box.minx + index, box.miny + index, box.minz), (1, 1, box.length))
            AREA_STRUCTURE.fill(ridge_box, palette.get_roof_block('bottom'), 2)
            AREA_STRUCTURE.fill(ridge_box.translate(dy=-1), palette.get_structure_block('z'), 4)

    def __gen_gable_cross(self, level, palette):
        for direction in Direction.cardinal_directions():
            if self[direction] is not None and isinstance(self[direction], _RoofSymbol):
                neighbour = self[direction]  # type: _RoofSymbol
                if abs(self._direction) == abs(direction) and abs(neighbour._direction.rotate()) == abs(direction):
                    box0 = self._box
                    box1 = neighbour._box
                    if direction in [Direction.North, Direction.South]:
                        box2 = TransformBox((box0.minx, box0.miny, box1.minz), (box0.width, box0.height, box1.length))
                    else:
                        box2 = TransformBox((box1.minx, box1.miny, box0.minz), (box1.width, box1.height, box0.length))
                    _RoofSymbol(box2, self._direction, self._roof_type).generate(level, palette=palette)


class _WallSymbol(Generator):
    def generate(self, level, height_map=None, palette=None):
        assert (self.width == 1 or self.length == 1)
        assert self.width > 0 and self.length > 0
        # assert (self.width * self.length >= 1)
        if self.length == 1:
            self._generate_xwall(level, palette)
        else:
            self._generate_zwall(level, palette)
        Generator.generate(self, level, height_map, palette)

    def _generate_xwall(self, level, palette):
        if self.width % 2 == 0:
            # even wall: split in two
            if self.width == 2:
                if bernouilli(0.5):
                    AREA_STRUCTURE.fill(self._box.expand(0, -1, 0), palette['window'], 7)
                AREA_STRUCTURE.fill(self._box, palette['wall'], 6)
            elif self.width == 4:
                AREA_STRUCTURE.fill(self._box, palette['wall'], 6)
                box_win = TransformBox(self._box.origin + (1, 0, 0), (2, self.height, 1))
                self.children.append(_WallSymbol(box_win))
            else:
                for half_wall_box in self._box.split(dx=random.randint(3, self.width - 3)):
                    self.children.append(_WallSymbol(half_wall_box))
        else:
            # uneven wall: derive in column | window | wall
            if self.width == 1:
                AREA_STRUCTURE.fill(self._box, palette['wall'], 6)
            else:
                box_col, box_wal = self._box.split(dx=2)
                box_win = TransformBox((self._box.origin + (1, 1, 0)), (1, self.height - 2, 1))
                AREA_STRUCTURE.fill(box_col, palette['wall'], 6)
                AREA_STRUCTURE.fill(box_win, palette['window'], 7)
                self.children.append(_WallSymbol(box_wal))

    def _generate_zwall(self, level, palette):
        if self.length % 2 == 0:
            # even wall: split in two
            if self.length == 2:
                if bernouilli(0.5):
                    AREA_STRUCTURE.fill(self._box.expand(0, -1, 0), palette['window'], 7)
                AREA_STRUCTURE.fill(self._box, palette['wall'], 6)
            elif self.length == 4:
                AREA_STRUCTURE.fill(self._box, palette['wall'], 6)
                box_win = TransformBox(self._box.origin + (0, 0, 1), (1, self.height, 2))
                self.children.append(_WallSymbol(box_win))
            else:
                for half_wall_box in self._box.split(dz=random.randint(3, self.length - 3)):
                    self.children.append(_WallSymbol(half_wall_box))
        else:
            # uneven wall: derive in column | window | wall
            if self.length == 1:
                AREA_STRUCTURE.fill(self._box, palette['wall'], 6)
            else:
                box_col, box_wal = self._box.split(dz=2)
                box_win = TransformBox((self._box.origin + (0, 1, 1)), (1, self.height - 2, 1))
                AREA_STRUCTURE.fill(box_col, palette['wall'], 6)
                AREA_STRUCTURE.fill(box_win, palette['window'], 7)
                self.children.append(_WallSymbol(box_wal))

    def generate_door(self, door_dir, door_x, door_z, level: worldLoader.WorldSlice, palette: HousePalette):
        dump()  # need to spawn windows to know where to place door(s)
        box = self._box
        entry = Point(door_x, door_z)
        if self.length > 1:
            is_win = [int(direct_interface.getBlock(box.minx, box.miny+1, box.minz+_).endswith(palette['window'])) for _ in range(box.length)]
            if sum(is_win) == 0: is_win = [0] + [1 for _ in range(len(is_win)-2)] + [0]
            door_val = [euclidean(entry, Point(box.minx, box.minz+_)) if is_win[_] or not sum(is_win) else 1000 for _ in range(box.length)]
            # door_z = choice(range(box.length), p=[1. * _ / sum(is_win) for _ in is_win])  # index position
            door_z = argmin([float(_) for _ in door_val])
            door_box = TransformBox(box.origin + (0, 0, door_z), (1, 3, 1))
            if door_z > 0 and is_win[door_z - 1]:
                door_box.expand(Direction.of(dz=-1), inplace=True)
            elif door_z < box.length - 1 and is_win[door_z + 1]:
                door_box.expand(Direction.of(dz=1), inplace=True)
            DoorGenerator(door_box, door_dir).generate(level, palette=palette)
        else:
            is_win = [int(direct_interface.getBlock(box.minx+_, box.miny+1, box.minz).endswith(palette['window'])) for _ in range(box.width)]
            if sum(is_win) == 0: is_win = [0] + [1 for _ in range(len(is_win)-2)] + [0]
            door_val = [euclidean(entry, Point(box.minx+_, box.minz)) if is_win[_] or not sum(is_win) else 1000. for _ in range(box.width)]
            door_x = argmin(door_val)
            door_box = TransformBox(box.origin + (door_x, 0, 0), (1, 3, 1))
            if door_x > 0 and is_win[door_x - 1]:
                door_box.expand(Direction.of(dx=-1), inplace=True)
            elif door_x < box.width - 1 and is_win[door_x + 1]:
                door_box.expand(Direction.of(dx=1), inplace=True)
            DoorGenerator(door_box, door_dir).generate(level, palette=palette)


class _BaseSymbol(Generator):
    def generate(self, level, height_map=None, palette=None):
        AREA_STRUCTURE.fill(self._box, palette['base'], 15)


class ProcHouseGeneratorBuilder():
    MIN_SIZE = 5
    AVG_SIZE = 7
    MAX_SIZE = 11
    MIN_RATIO = 2/3

    def __init__(self, origin: Position, mask: np.ndarray):
        self.__origin = origin
        self.__mask = mask

    def build(self, box: BoundingBox, n_iter: int = 100) -> _RoomSymbol:
        best_foot_print: _RoomSymbol = None
        best_score: int = 0
        for _ in range(n_iter):
            room: _RoomSymbol = self.gen_solution(box)
            if room is None:
                return None
            elif self.is_feasible(room) and self.score(room) > best_score:
                best_foot_print = room
                best_score = self.score(room)

        return best_foot_print

    def gen_solution(self, box: BoundingBox) -> Union[_RoomSymbol, None]:
        MIN_S, MAX_S, MIN_R, MAX_R = self.MIN_SIZE, self.MAX_SIZE, self.MIN_RATIO, 1. / self.MIN_RATIO
        AVG_S = self.AVG_SIZE

        # First, apply size constraints to the house
        if box.width < self.MIN_SIZE or box.length < self.MIN_SIZE:
            return None

        # Select random feasible rectangle in the box
        rx = rz = rw = rl = 0  # rx, rz, width and length of the main room
        while not (MIN_S <= rw <= MAX_S and MIN_S <= rl <= MAX_S and MIN_R <= (rw / rl) <= MAX_R):
            try:
                rw = random.randint(AVG_S, box.width) if box.width > AVG_S else box.width
                rl = random.randint(AVG_S, box.length) if box.length > AVG_S else box.length
                rx = random.randint(0, box.width - rw)
                rz = random.randint(0, box.length - rl)
            except ValueError:
                continue
        main_box = TransformBox(box.origin + (rx, 1, rz), (rw, box.height, rl))
        main_room = _RoomSymbol(main_box, has_base=True)

        try:
            a1w = random.randint(MIN_S, rw - 2)
            a1x = rx + random.randint(0, rw - a1w)
            max_a1l = min(MAX_S, int(a1w * MAX_R))
            a1z = random.choice([z for z in range(box.length) if z < rz or (rz + rl) < z < min(box.maxz, rz + rl + max_a1l)])
            if a1z < rz:
                a1l = rz - a1z + 1
            else:
                a1l = a1z - rz - rl
                a1z = rz + rl - 1
            a1box = TransformBox(box.origin + (a1x, 1, a1z), (a1w, box.height - 2, a1l))
            main_room[Direction.of(dz=(a1z - rz))] = _RoomSymbol(a1box, has_base=True)

        except (ValueError, IndexError):
            pass

        try:
            a2l = random.randint(MIN_S, rl - 2)
            a2z = rz + random.randint(0, rl - a2l)
            max_a2w = min(MAX_S, int(a2l * MAX_R))
            a2x = random.choice([x for x in range(box.width) if x < rx or (rx + rw) < x < min(box.maxx, rx + rw + max_a2w)])
            if a2x < rx:
                a2w = rx - a2x + 1
            else:
                a2w = a2x - rx - rl
                a2x = rx + rw - 1
            a2box = TransformBox(box.origin + (a2x, 1, a2z), (a2w, box.height - 2, a2l))
            main_room[Direction.of(dx=(a2x - rx))] = _RoomSymbol(a2box, has_base=True)

        except (ValueError, IndexError):
            pass

        return main_room

    def is_feasible(self, room: _RoomSymbol) -> bool:
        """
        Compares solution to parcel mask
        :param room:
        :return:
        """
        rooms: List[_RoomSymbol] = [room] + room.children
        for sub_room in rooms:
            min_x = sub_room.origin.x - self.__origin.x
            min_z = sub_room.origin.z - self.__origin.z
            if not self.__mask[min_x:(min_x + sub_room.width), min_z:(min_z + sub_room.length)].all():
                return False
        return True

    @staticmethod
    def score(room: _RoomSymbol) -> int:
        """
        Score of the solution, equals the total surface of the foot print
        :param room: feasible solution
        :return: surface of the solution
        """
        return room.surface * (sum(r.surface for r in room.children) if room.children else 1)
