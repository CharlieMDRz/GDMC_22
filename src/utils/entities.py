from typing import Tuple, List, Dict

from gdpc import direct_interface, worldLoader

from .geometry_utils import Direction


__all__ = ['Entity', 'detect_entities']


class Entity:
    """
    Represents a single MC entity and holds several helper functions to add, delete or move entities in the build area
    """
    entity_type: str  # MC entity type, eg "minecraft:zombie"
    position: Tuple[int, int, int] = None  # Current position
    __init_pos: Tuple[int, int, int] = None  # Initial position, None if summoned during generation
    __summoned: bool = False  # Whether entity has been summoned, sets to True if an initial position is specified
    __special_args: Dict[str, str]  # Special MC arguments for entities

    def __init__(self, _type: str, initial_pos: Tuple[int, int, int] = None, name: str = "", **entity_args):
        """
        Instantiates a new entity, either existent or non-existent. This class will assume the entity exists already iff
        the parameter initial_pos is specified
        :param _type: MC entity type
        :param initial_pos: X Y Z position of the entity
        :param name: optional name for an entity
        :param entity_args: other entity optional args, listed at https://minecraft.fandom.com/wiki/Entity_format
        """
        self.entity_type = _type
        if initial_pos:
            self.position = initial_pos
            self.__init_pos = initial_pos
            self.__summoned = True
        self.__special_args = {arg: str(val) for arg, val in entity_args.items()}
        if name:
            self.set_name(name)

    def __setitem__(self, property, value):
        self.__special_args[property] = value

    def set_name(self, name: str):
        """
        Names the entity, ie adds the property: {CustomName: '{"text": "CatName"}'}
        :param name: name
        """
        self["CustomName"] = '\'{"text": "' + name + '"}\''

    def __str__(self):
        return "type={}, x={}, y={}, z={}".format(self.entity_type, *self.position)

    @property
    def nbt_str(self):
        """
        :return: Formatted special args
        """
        return ', '.join(f'{arg}: {val}' for arg, val in self.__special_args.items())

    def move_to(self, dest: Tuple[int, int, int]) -> None:
        """
        Moves current entity to the specified location. Summons entity if it does not exist yes
        :param dest: X Y Z destination
        :return:
        """
        if self.__summoned:
            # TPs existing entity
            cmd = "tp @e[{}, limit=1, sort=nearest] {} {} {}".format(str(self), *dest)
        else:
            # Summons non-existent entity
            cmd = "summon {} {} {} {} ".format(self.entity_type, *dest) + '{' + self.nbt_str + '}'
        self.__summoned = True
        self.position = dest
        return direct_interface.runCommand(cmd)

    def kill(self) -> None:
        """
        Kills an entity
        :return:
        """
        if self.__summoned:
            self.__summoned = False
            cmd = f"kill @e[{str(self)}, limit=1, sort=nearest]"
            return direct_interface.runCommand(cmd)

    @staticmethod
    def get_orientation(direction):
        return {
            Direction.Bottom: 0,
            Direction.Top: 1,
            Direction.North: 2,
            Direction.South: 3,
            Direction.West: 4,
            Direction.East: 5
        }[direction]

    def reset(self) -> None:
        """
        Resets current entity to its initial state
        """
        if self.__init_pos:
            self.move_to(self.__init_pos)
        else:
            self.kill()


def detect_entities(level: worldLoader.WorldSlice) -> List[Entity]:
    """
    Detect world entities from detethe nbt file of the build area
    :param level: world slice
    :return: list of entities
    """
    entities = []

    for chunk in level.nbtfile['Chunks']:
        for tag_entity in chunk['Level']['Entities']:
            e_id = tag_entity['id'].value
            pos = tuple(_.value for _ in tag_entity['Pos'])
            entities.append(Entity(e_id, pos))

    return entities


def get_item_frame_entity(item: str, facing: Direction = Direction.Top, invisible: bool = False, item_rotation: int = 0, **kwargs) -> Entity:

    entity_args = {
        "Facing": Entity.get_orientation(facing),
        "Item": '{id:"' f'{item}' '", Count:1}',
        "Invisible": int(invisible),
        "ItemRotation": item_rotation
    }

    kwargs.update(entity_args)

    from terrain import EntityManager
    return EntityManager().get_entity("item_frame", **kwargs)
