from typing import List, Dict

from gdpc import worldLoader

from utils import Position, euclidean, Point
from utils.entities import *


class EntityManager:
    WILD_ANIMALS = {'bat', 'bee', 'fox', 'llama', 'ocelot', 'panda', 'polar_bear', 'turtle'}
    FARMING_ANIMALS = {'chicken', 'cow', 'donkey', 'horse', 'mule', 'pig', 'sheep'}
    TAMED_ANIMALS = {'cat', 'parrot', 'rabbit', 'wolf'}
    FISH = {'cod', 'dolphin', 'pufferfish', 'salmon', 'squid', 'tropical_fish'}

    __entities: Dict[str, List[Entity]]
    __is_wild: Dict[Entity, bool]

    def __init__(self, existing_entities):
        self.__entities = {}
        self.__is_wild = {}
        for e in existing_entities:
            self.__add_entity(e, True)

    @classmethod
    def from_world_slice(cls, world_slice: worldLoader.WorldSlice):
        return EntityManager(detect_entities(world_slice))

    def __add_entity(self, entity: Entity, is_wild: bool) -> Entity:
        etype = entity.entity_type.replace("minecraft:", "")
        self.__entities.setdefault(etype, []).append(entity)
        self.__is_wild[entity] = is_wild
        return entity

    def get_entity(self, e_type, position: Position = None, **entity_args) -> Entity:
        """
        Gets a wild entity of the specified type, or creates one otherwise
        :param e_type: type of the entity to return
        :param position: if specified, will return the closest entity
        """
        def weight(e: Entity):
            x0, _, z0 = e.position
            x1, z1 = position.abs_x,  position.abs_z
            return euclidean(Point(x0, z0), Point(x1, z1))

        if e_type in self.__entities:
            entities = [_ for _ in self.__entities[e_type] if self.__is_wild[_]]

            if position:
                entities = sorted(entities, key=weight)

            if entities:
                e: Entity = entities[0]
                self.__is_wild[e] = False
                return e

        return self.__add_entity(Entity(e_type, **entity_args), False)

    @property
    def wild_entity_population(self):
        return {etype: sum(1 for e in elist if self.__is_wild[e]) for etype, elist in self.__entities.items()}

    def reset(self):
        for entity_list in self.__entities.values():
            for entity in entity_list:
                entity.reset()


def get_most_populated_animal(entity_manager: EntityManager):
    count = entity_manager.wild_entity_population
    animals = set(count.keys()).intersection(entity_manager.FARMING_ANIMALS)
    try:
        return sorted(animals, key=count.__getitem__)[-1]
    except IndexError:
        return None
