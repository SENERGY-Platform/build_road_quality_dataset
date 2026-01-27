
# PROBLEM: No actual clear compatibility between OSM labels (smoothness & surface) and our Street labels (good | medium | bad)
# GOAL: Provide different mappings of these OMS labels to Street labels with different data sets
# GOAL FORMAT PER DATASET:
    # timestamp | lat | lon | label | VehicleType | speed | rawVib | MildScore | StandardScore | StrictScore | ...

# Smoothness options:
    # for intermediate try:
    # (1) intermediate = 1.0
    # (2) intermediate = 1.5
SMOOTHNESS_SCENARIOS = ['sm1', 'sm2']

# Surface options:
    # (1) use specific continuous numbers
    # (2) build discrete points and 1.5 for duplicates
    # (3) two labeled points (diff. labels)
    # decide for one of the duplicates (prob. not)
SURFACE_SCENARIOS = ['surf1', 'surf2', 'surf3']

# Combination options:
    # (1) prio smoothness
    # (2) prio road type
    # (3) use avg of both sub values
    # (4) add both smoothness and duplicate surface point+labels
COMBINATION_SCENARIOS = ['c1', 'c2', 'c3', 'c4']

# --------------------------------------------------------------------------------------------------------------------
# goal labels

STREET_LABEL_TO_NUM = {
    "good": 0,
    "medium": 1,
    "bad": 2
}

# ----------------------------------------------------------------------------------------------------------------------
# smoothness mappings

SMOOTHNESS_TO_LABEL_1 = {
    "excellent": 0.0,
    "good": 0.0,
    "bad": 2.0,
    "very bad": 2.0,
    "very_bad": 2.0, # found as labeling mistake in api data
    "horrible": 2.0,
    "very horrible": 2.0,
    'intermediate': 1.0
}
SMOOTHNESS_TO_LABEL_2 = SMOOTHNESS_TO_LABEL_1.copy()
SMOOTHNESS_TO_LABEL_2.update({'intermediate': 1.5})

# ----------------------------------------------------------------------------------------------------------------------
# surface mappings

SURFACE_TO_LABEL_1 = {
    # paved
    # 'paved': None,
    # 'ground': None,
    'asphalt': 0.3,
    'concrete': 1.2,
    'concrete:lanes': 1.2,
    'concrete:plates': 1.2,
    'paving_stones': 1.1,
    'paving_stones:lanes': 1.1,
    'sett': 1.8,
    'unhewn_cobblestone': 2.0,
    'cobblestone': 2.0,
    'bricks': 1.6,
    'wood': 1.5,
    'grass_paver': 2.0,
    # unpaved
    'dirt': 1.2,
    'compacted': 1.0,
    'ground;mud': 1.7,
    'fine_gravel': 1.5,
    'gravel': 2.0,
    'shells': 2.0,
    'rock': 2.0,
    'pebblestone': 2.0,
    'sand': 2.0,
    'mud': 2.0,
    'grass': 2.0,
}

SURFACE_TO_LABEL_2 = {
    # 'paved': None,
    # 'ground': None,
    'asphalt': 0.5,
    'concrete': 1.5,
    'concrete:lanes': 1.5,
    'concrete:plates': 1.5,
    'paving_stones': 1.5,
    'paving_stones:lanes': 1.5,
    'sett': 1.5,
    'bricks': 1.5,
    'wood': 1.5,
    'compacted': 1.0,
    'fine_gravel': 1.5,
    'dirt': 1.5,
    'ground;mud': 1.5,
    'unhewn_cobblestone': 2.0,
    'cobblestone': 2.0,
    'grass_paver': 2.0,
    'gravel': 2.0,
    'shells': 2.0,
    'rock': 2.0,
    'pebblestone': 2.0,
    'sand': 2.0,
    'mud': 2.0,
    'grass': 2.0,
}

SURFACE_TO_LABEL_3 = {
    # 'paved': None,
    # 'ground': None,
    'asphalt': [0.0, 1.0],
    'concrete': [1.0, 2.0],
    'concrete:lanes': [1.0, 2.0],
    'concrete:plates': [1.0, 2.0],
    'paving_stones': [1.0, 2.0],
    'paving_stones:lanes': [1.0, 2.0],
    'wood': [1.0, 2.0],
    'sett': [1.0, 2.0],
    'dirt': [1.0, 2.0],
    'bricks': [1.0, 2.0],
    'ground;mud': [1.0, 2.0],
    'compacted': [1.0, 2.0],
    'fine_gravel': [1.0, 2.0],
    'unhewn_cobblestone': [2.0],
    'cobblestone': [2.0],
    'grass_paver': [2.0],
    'gravel': [2.0],
    'shells': [2.0],
    'rock': [2.0],
    'pebblestone': [2.0],
    'sand': [2.0],
    'mud': [2.0],
    'grass': [2.0],
}

# ----------------------------------------------------------------------------------------------------------------------

# helper
medium_road_types = {
    # paved
    'concrete:lanes',       # duplicate
    'concrete:plates',      # duplicate
    'paving_stones',        # duplicate
    'paving_stones:lanes',  # duplicate
    'sett',                 # duplicate
    'bricks',
    # unpaved
    'compacted',
    'fine_gravel'           # duplicate
}
bad_road_types = {
    # paved
    'concrete:lanes',       # duplicate
    'concrete:plates',      # duplicate
    'paving_stones',        # duplicate
    'paving_stones:lanes',  # duplicate
    'sett',                 # duplicate
    'bricks',               # duplicate
    'grass_paver',
    'unhewn_cobblestone',
    'cobblestone',
    # unpaved
    'fine_gravel',          # duplicate
    'gravel',
    'shells',
    'rock',
    'pebblestone',
    'sand',
    'mud',
    'grass',
}
PAVEDISH = {
    "asphalt",
    "concrete",
    "paving_stones",
    "cobblestone",
    "sett",
    "concrete:lanes",
    "concrete:plates",
}
UNPAVEDISH = {
    "gravel",
    "fine_gravel",  # often unpaved in practice
    "compacted",
    "ground",
    "dirt",
    "earth",
    "sand",
    "mud",
    "grass",
}