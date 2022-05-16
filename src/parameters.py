# Road network
MAX_LAMBDA = 15
MAX_ROAD_WIDTH = 5
MIN_ROAD_WIDTH = 3
DIST_BETWEEN_NODES = 12

# Road cycles
MAX_DISTANCE_CYCLE = 30
MIN_DISTANCE_CYCLE = 10
MIN_CYCLE_GAIN = 2.5
CYCLE_ALTERNATIVES = 10

MAX_WATER_EXPLORATION = 100
MAX_LAVA_EXPLORATION = 50
PARCEL_REUSE_ADVANTAGE = 2.5  # interest multiplying cost when evaluating a parcel type replacement
MAX_POND_EXPLORATION = 144  # should ideally be higher than the largest ponds to exterminate all of them

# Path finding costs parameters
BRIDGE_COST = 10
BRIDGE_UNIT_COST = 5
RAIL_ROAD_SPACING = 6
RAIL_ROAD_PENALTY = 7

# buildings dimensions
MAX_HEIGHT = 16  # 12 -> max 3 floors
BUILDING_HEIGHT_SPREAD = 40  # at that distance from the center, buildings' height is a third of max height (1/e ~ .36)
AVERAGE_PARCEL_SIZE = 12
MAX_PARCELS_IN_BLOCK = 6
MIN_PARCEL_SIZE = 1
MIN_RATIO_SIDE = .7

# distances
MIN_DIST_TO_LAVA = 10
MIN_DIST_TO_OCEAN = 8
MIN_DIST_TO_RIVER = 4

# seeding
SEED_COUNT = 200
REPLACE_PARCEL_TYPE_EXPLORATION = 15

MAX_INT = 1 << 15
MAX_FLOAT = float(MAX_INT)
TERRAFORM_ITERATIONS = 3

PALETTE_MUTATION_PROBABILITY = .4

ACTIVATED_WINDMILLS = 0
