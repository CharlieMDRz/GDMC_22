from utils import BoundingBox, Position, Point
from numpy import zeros, int32
from utils.block_utils import setBlock, BuildArea


class Structure(BoundingBox):
    def __init__(self, origin, size):
        super().__init__(origin, size)
        self.__priority = zeros(size, dtype=int32)

    @classmethod
    def from_box(cls, box: BoundingBox):
        return cls(box.origin, box.size)

    def set(self, pos: Position, blockstate: str, priority: int = 1):
        if isinstance(pos, Position):
            absolute_point = Point(pos.abs_x, pos.abs_z, pos.y)
            relative_point = pos
        else:
            pos: Point
            absolute_point = pos
            relative_point = pos - self.origin
        if absolute_point.coords not in self:
            return
        prev_priority = self.__priority[relative_point.coords]

        if priority > prev_priority:
            setBlock(absolute_point, blockstate)
            self.__priority[relative_point.coords] = priority

    def fill(self, box: BoundingBox, blockstate: str, priority: int = 1):
        for x, y, z, in box.positions:
            p = Point(x, z, y)
            self.set(p, blockstate, priority)

    @property
    def origin(self) -> Position:
        return Position(self.minx, self.minz, self.miny)


AREA_STRUCTURE: Structure = Structure((BuildArea().x, 0, BuildArea().z), (BuildArea().width, 256, BuildArea().length))
