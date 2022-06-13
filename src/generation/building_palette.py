import random
from typing import Set

from gdpc import lookup

from utils import BlockAPI
from utils.block_utils import random_block, random_material

b = BlockAPI.blocks


class HousePalette(dict):
    __material_main: str  # roofs
    __material_alt: str  # under roofs, inside stairs
    __wall_block_alt: str  # roofed wall

    def __init__(self, base_bloc, floor_block, struct_block, wall_block, window_block, roof_type, roof_block, door_mat,
                 roof_alt=None, wall_alt=None):
        super(HousePalette, self).__init__()
        self['roofType'] = roof_type
        self['roofBlock'] = roof_block.replace('_stairs', '')
        self['base'] = base_bloc
        self['structure'] = struct_block if struct_block is not None else wall_block
        self['wall'] = wall_block
        self['window'] = window_block
        self['floor'] = floor_block
        self['door'] = door_mat
        self['roofAlt'] = roof_alt if roof_alt else roof_block
        self['wallAlt'] = wall_alt if wall_alt else wall_block

    def get_roof_block(self, facing, direction=None, alternate=None):
        if direction is None:
            roof_block = BlockAPI.getSlab(self['roofBlock'], type=facing)
        elif self['roofType'] == 'flat':
            roof_block = self['roofBlock']
        else:
            if alternate is None:
                alternate = (facing == 'top')
            roof_block = BlockAPI.getStairs(self['roofAlt'] if alternate else self['roofBlock'], half=facing,
                                            facing=direction)
        return roof_block

    def get_structure_block(self, axis):
        block = self['structure'].replace('minecraft:', '')
        from utils.block_utils import BlockStateDict
        blockstates = BlockStateDict()
        if 'axis' in blockstates[block]:
            return f"{block}[axis={axis}]"
        return self['structure']

    @classmethod
    def mutant(cls, palette):
        mutant_palette = HousePalette(*palette.values())
        for k, v in palette.items():
            mutant_palette[k] = v

        mutated_key = random.choice(list(mutant_palette.keys()))
        if mutated_key == 'roofType': return palette
        if mutated_key == 'roofBlock':
            mutant_palette[mutated_key] = random_material(lookup.SLABS, lookup.STAIRS) if palette[
                                                                                              'roofType'] == 'gable' else random_block()
        elif mutated_key == 'roofAlt':
            mutant_palette[mutated_key] = random_material(lookup.STAIRS)
        elif mutated_key == 'door':
            mutant_palette[mutated_key] = random_material(lookup.DOORS, lookup.FENCES, lookup.GATES)
        elif mutated_key == 'window':
            mutant_palette[mutated_key] = random.choice(lookup.GLASS)
        else:
            mutant_palette[mutated_key] = random_block()

        return mutant_palette


class HousePaletteGenerator:
    def __init__(self, base, floor, frame, walls, windows, roof_type, roof, doors, roof_alt=None, wall_alt=None):
        self.base_block = base
        self.floor_block = floor
        self.frame_block = frame
        self.wall_block = walls
        self.wall_block_alt = wall_alt
        self.windows_block = windows
        self.roof_block = roof
        self.roof_type = roof_type
        self.roof_block_alt = roof_alt
        self.door_material = doors

    def __call__(self, *args, **kwargs):
        def norm(arg):
            if arg is None or type(arg) is str:
                return arg
            else:
                return random.choice(list(arg))

        palette_args = (
            self.base_block, self.floor_block, self.frame_block, self.wall_block,
            self.windows_block, self.roof_type, self.roof_block, self.door_material,
            self.roof_block_alt, self.wall_block_alt
        )

        return HousePalette(*map(norm, palette_args))


class PaletteGenerators:
    PARIS = HousePaletteGenerator(
        b.Stone, b.SprucePlanks, b.SmoothSandstone, b.BirchPlanks, b.WhiteStainedGlassPane, 'gable',
        [b.PolishedBlackstoneBrickStairs, b.BlackstoneStairs, b.PolishedBlackstoneStairs], 'birch', b.SmoothSandstone,
        b.ChiseledSandstone)

    LYON = HousePaletteGenerator(
        b.Stone, b.OakPlanks, b.StrippedOakLog, [b.OrangeTerracotta, b.Terracotta, b.YellowTerracotta], b.GlassPane,
        'gable', [b.PolishedGraniteStairs, b.GraniteStairs], 'oak', b.OakStairs)

    BORDEAUX = HousePaletteGenerator(
        b.Stone, b.OakPlanks, [None, b.StrippedOakLog, b.CutSandstone], [b.OakPlanks, b.StrippedBirchLog], b.LightGrayStainedGlassPane,
        'gable', [b.StoneBrickStairs, b.AndesiteStairs, b.PolishedAndesiteStairs], 'oak', b.OakStairs)

    MARSEILLE = HousePaletteGenerator(
        b.Stone, b.CutSandstone, b.QuartzPillar, [b.BirchPlanks, b.StrippedBirchWood, b.StrippedJungleWood, b.YellowTerracotta], b.LightBlueStainedGlassPane,
        'gable', [b.AcaciaStairs, b.BrickStairs], 'jungle', b.QuartzStairs)

    STRASBOURG = HousePaletteGenerator(
        b.Stone, b.OakPlanks, b.SpruceLog, [b.WhiteConcrete, b.WhiteWool, b.Terracotta, b.SmoothSandstone], b.GlassPane,
        'gable', [b.NetherBrickStairs, b.DarkOakStairs, b.SpruceStairs], 'oak', b.CrimsonStairs)

    LILLE = HousePaletteGenerator(
        b.Stone, b.DarkOakPlanks, [b.StrippedJungleLog, b.StrippedOakLog], [b.AcaciaPlanks, b.StrippedAcaciaLog, b.Bricks, b.Terracotta], b.GrayStainedGlassPane,
        'gable', [b.CobblestoneStairs, b.MossyCobblestoneStairs, b.PolishedBlackstoneBrickStairs], 'acacia', b.SmoothSandstoneStairs)

    BAYONNE = HousePaletteGenerator(
        b.Stone, b.DarkOakPlanks, [b.AcaciaPlanks, b.QuartzPillar, None], [b.WhiteTerracotta, b.LightGrayTerracotta], b.AcaciaFence,
        'gable', [b.DarkPrismarineStairs, b.WarpedStairs], 'acacia', b.SmoothSandstoneStairs)

    BASIC_OAK = HousePaletteGenerator(
        [b.Cobblestone, b.Stone], b.SprucePlanks, [b.OakLog, b.StrippedOakLog, b.StrippedSpruceLog], b.OakPlanks,
        b.LightGrayStainedGlassPane, 'gable', [b.StoneBrick, 'oak', 'spruce'], 'oak', [None, b.StoneBrick, 'oak', 'spruce'],
        [None, None, None, b.BoneBlock])

    BASIC_DARK_OAK = HousePaletteGenerator(
        b.Cobblestone, b.SprucePlanks, b.DarkOakLog, b.DarkOakPlanks,
        b.WhiteStainedGlassPane, 'gable', b.StoneBrick, 'dark_oak')

    BASIC_SPRUCE = HousePaletteGenerator(
        [b.Cobblestone, b.Stone, b.MossyCobblestone], b.OakPlanks, [b.SpruceLog, b.StrippedSpruceLog, b.StoneBricks], b.SprucePlanks,
        b.LightGrayStainedGlassPane, 'gable', [b.StoneBrickStairs, b.MossyStoneBrickStairs, 'spruce'], 'spruce', [None, 'spruce'],
        [None, None, None, b.BoneBlock])

    # unintended swear below oopsie
    BASIC_BIRCH = HousePaletteGenerator(
        [b.Cobblestone, b.Stone], b.OakPlanks, [b.BirchLog, b.StrippedBirchLog, b.StrippedOakLog], b.BirchPlanks,
        b.WhiteStainedGlassPane, 'gable', [b.StoneBrick, 'birch', 'oak'], 'birch', [None, b.StoneBrick, 'oak', 'birch'],
        [None, None, None, b.BoneBlock])

    BASIC_ACACIA = HousePaletteGenerator(
        b.Cobblestone, b.BirchPlanks, b.AcaciaLog, b.AcaciaPlanks,
        b.WhiteStainedGlassPane, 'gable', b.StoneBrick, 'acacia')

    BASIC_JUNGLE = HousePaletteGenerator(
        b.Cobblestone, b.OakPlanks, b.JungleLog, b.JunglePlanks,
        b.WhiteStainedGlassPane, 'gable', b.StoneBrick, 'jungle')

    BASIC_SAND = HousePaletteGenerator(
        b.Cobblestone, b.SprucePlanks, [b.SmoothSandstone, b.CutSandstone], [b.Sandstone]*3 + [b.Terracotta, b.GreenTerracotta],
        b.BirchFence, 'flat', b.ChiseledSandstone, 'oak', roof_alt=b.Sandstone)

    BASIC_RED_SAND = HousePaletteGenerator(
        b.Cobblestone, b.SprucePlanks, b.SmoothRedSandstone, b.RedSandstone,
        b.BirchFence, 'flat', b.ChiseledRedSandstone, 'oak')

    BASIC_TERRACOTTA = HousePaletteGenerator(
        b.RedSandstone, b.SprucePlanks, b.OakLog, b.Terracotta,
        b.SpruceFence, 'flat', b.Terracotta, 'oak')

    BUNKER = HousePaletteGenerator(
        b.Stone, b.SmoothStone, [b.StoneBricks, b.CrackedStoneBricks], [b.Cobblestone, b.InfestedCobblestone],
        'iron_bars', 'flat', b.PolishedAndesiteSlab, b.IronDoor)


PG = PaletteGenerators

stony_palette = {"cobblestone": 0.7, "gravel": 0.2, "stone": 0.1}
# Defines for each biome the acceptable palettes. Adapted from pymclevel.biome_types

city_palette_gens = [PG.PARIS, PG.MARSEILLE, PG.LYON, PG.BORDEAUX, PG.STRASBOURG, PG.LILLE, PG.BAYONNE]

biome_palettes = {
    'badlands': city_palette_gens + [PG.BASIC_TERRACOTTA],
    'badlands_plateau': city_palette_gens + [PG.BASIC_TERRACOTTA],
    'bamboo_jungle': city_palette_gens + [PG.BASIC_JUNGLE],
    'bamboo_jungle_hills': city_palette_gens + [PG.BASIC_JUNGLE],
    # 'basalt_deltas': city_palette_gens + [],
    'beach': city_palette_gens + [PG.BASIC_OAK],
    'birch_forest': city_palette_gens + [PG.BASIC_BIRCH],
    'birch_forest_hills': city_palette_gens + [PG.BASIC_BIRCH],
    # 'cold_ocean': city_palette_gens + [],
    # 'crimson_forest': city_palette_gens + [],
    'dark_forest': city_palette_gens + [PG.BASIC_DARK_OAK],
    'dark_forest_hills': city_palette_gens + [PG.BASIC_DARK_OAK],
    # 'deep_cold_ocean': city_palette_gens + [],
    # 'deep_frozen_ocean': city_palette_gens + [],
    # 'deep_lukewarm_ocean': city_palette_gens + [],
    # 'deep_ocean': city_palette_gens + [],
    # 'deep_warm_ocean': city_palette_gens + [],
    'desert': city_palette_gens + [PG.BASIC_SAND],
    'desert_hills': city_palette_gens + [PG.BASIC_SAND],
    'desert_lakes': city_palette_gens + [PG.BASIC_SAND],
    # 'end_barrens': city_palette_gens + [],
    # 'end_highlands': city_palette_gens + [],
    # 'end_midlands': city_palette_gens + [],
    'eroded_badlands': city_palette_gens + [PG.BASIC_TERRACOTTA],
    'flower_forest': city_palette_gens + [PG.BASIC_OAK, PG.BASIC_BIRCH],
    'forest': city_palette_gens + [PG.BASIC_OAK, PG.BASIC_BIRCH],
    # 'frozen_ocean': city_palette_gens + [],
    # 'frozen_river': city_palette_gens + [],
    'giant_spruce_taiga': city_palette_gens + [PG.BASIC_SPRUCE],
    'giant_spruce_taiga_hills': city_palette_gens + [PG.BASIC_SPRUCE],
    'giant_tree_taiga': city_palette_gens + [PG.BASIC_SPRUCE],
    'giant_tree_taiga_hills': city_palette_gens + [PG.BASIC_SPRUCE],
    'gravelly_mountains': city_palette_gens + [PG.BASIC_SPRUCE],
    'ice_spikes': city_palette_gens + [PG.BUNKER],
    'jungle': city_palette_gens + [PG.BASIC_JUNGLE],
    'jungle_edge': city_palette_gens + [PG.BASIC_JUNGLE],
    'jungle_hills': city_palette_gens + [PG.BASIC_JUNGLE],
    # 'lukewarm_ocean': city_palette_gens + [],
    'modified_badlands_plateau': city_palette_gens + [PG.BASIC_TERRACOTTA],
    'modified_gravelly_mountains': city_palette_gens + [PG.BASIC_SPRUCE, ],
    'modified_jungle': city_palette_gens + [PG.BASIC_JUNGLE],
    'modified_jungle_edge': city_palette_gens + [PG.BASIC_JUNGLE],
    'modified_wooded_badlands_plateau': city_palette_gens + [PG.BASIC_DARK_OAK],
    'mountain_edge': city_palette_gens + [PG.BASIC_SPRUCE, PG.BUNKER],
    'mountains': city_palette_gens + [PG.BASIC_SPRUCE, PG.BUNKER],
    'mushroom_field_shore': city_palette_gens + [],
    'mushroom_fields': city_palette_gens + [],
    # 'nether_wastes': city_palette_gens + [],
    # 'ocean': city_palette_gens + [],
    'plains': city_palette_gens + [PG.BASIC_OAK],
    'river': city_palette_gens + [PG.BASIC_OAK],
    'savanna': city_palette_gens + [PG.BASIC_ACACIA],
    'savanna_plateau': city_palette_gens + [PG.BASIC_ACACIA],
    'shattered_savanna': city_palette_gens + [PG.BASIC_ACACIA],
    'shattered_savanna_plateau': city_palette_gens + [PG.BASIC_ACACIA],
    # 'small_end_islands': city_palette_gens + [],
    'snowy_beach': city_palette_gens + [PG.BASIC_OAK],
    'snowy_mountains': city_palette_gens + [PG.BASIC_SPRUCE, PG.BUNKER],
    'snowy_taiga': city_palette_gens + [PG.BASIC_SPRUCE],
    'snowy_taiga_hills': city_palette_gens + [PG.BASIC_SPRUCE],
    'snowy_taiga_mountains': city_palette_gens + [PG.BASIC_SPRUCE],
    'snowy_tundra': city_palette_gens + [PG.BUNKER],
    # 'soul_sand_valley': city_palette_gens + [],
    'stone_shore': city_palette_gens + [PG.BASIC_OAK],
    'sunflower_plains': city_palette_gens + [PG.BASIC_OAK],
    'swamp': city_palette_gens + [PG.BASIC_OAK, PG.BASIC_SPRUCE],
    'swamp_hills': city_palette_gens + [PG.BASIC_SPRUCE, PG.BASIC_OAK],
    'taiga': city_palette_gens + [PG.BASIC_SPRUCE],
    'taiga_hills': city_palette_gens + [PG.BASIC_SPRUCE],
    'taiga_mountains': city_palette_gens + [PG.BASIC_SPRUCE],
    'tall_birch_forest': city_palette_gens + [PG.BASIC_OAK, PG.BASIC_BIRCH],
    'tall_birch_hills': city_palette_gens + [PG.BASIC_OAK, PG.BASIC_BIRCH],
    # 'the_end': city_palette_gens + [],
    # 'the_void': city_palette_gens + [],
    # 'warm_ocean': city_palette_gens + [],
    # 'warped_forest': city_palette_gens + [],
    'wooded_badlands_plateau': city_palette_gens + [PG.BASIC_TERRACOTTA],
    'wooded_hills': city_palette_gens + [PG.BASIC_OAK],
    'wooded_mountains': city_palette_gens + [PG.BASIC_SPRUCE, PG.BASIC_OAK]
}

unused_palettes = set(
    filter(
        lambda attr: isinstance(attr, HousePaletteGenerator),
        map(
            lambda attr_name: getattr(PaletteGenerators, attr_name),
            vars(PaletteGenerators)
        )
    )
)


def get_biome_palette(biome) -> HousePaletteGenerator:
    result: HousePaletteGenerator
    try:
        palette_options: Set[HousePaletteGenerator] = unused_palettes.intersection(biome_palettes[biome])
        if palette_options:
            result = palette_options.pop()
        else:
            result = random.choice(list(unused_palettes))
    except Exception:
        print("Exception occurred when getting palette for biome: {}".format(biome))
        result = random.choice(list(unused_palettes))
    unused_palettes.remove(result)
    return result


def random_palette() -> HousePalette:
    """
    Generates a random palette, mostly for debug purposes, could be nice to have a cool palette generator
    :return: block palette to generate buildings
    """
    roof_type = random.choice(['gable'] * 3 + ['flat'])
    roof_block = random_material(lookup.SLABS, lookup.STAIRS) if roof_type == 'gable' else random_block()
    return HousePalette(
        random_block(),
        random_block(),
        random_block(),
        random_block(),
        random.choice(lookup.GLASS),
        roof_type,
        roof_block,
        random_material(lookup.DOORS, lookup.FENCES, lookup.GATES),
        random_material(lookup.STAIRS),
        random_block()
    )
