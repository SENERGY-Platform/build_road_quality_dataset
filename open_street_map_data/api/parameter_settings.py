# settings
# api
EARTH_RADIUS_M = 6371000.0
API_REQUEST_DISTANCE_M = 20
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# mappings
# 3 lane motorways are up to 12m wide, oms element will be in the middle, so 5-6m margin to reach from outer lane.
# important: filter pathways, walking and standing by the side of a street out in a different way (i e. vehicle type, speed min etc.)
MAX_POINT_DISTANCE_M = 6

# within 10cm two elements are equally close
CLOSEST_WAYS_MARGIN_M = 0.10


# OSM highway=* values grouped for "car-drivable street" filtering (from Key:highway wiki page)
HIGHWAY_ALLOWED_CAR_STREETS = [
    # Roads
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",

    # Link roads
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",

    # Special road types that are still generally car-drivable streets/roads
    "living_street",
    "service",
    "track",
    "passing_place",
]

HIGHWAY_FORBIDDEN_NOT_CAR_STREETS = [
    # Special road types that are not “normal car streets”
    "pedestrian",
    "bus_guideway",
    "raceway",
    "escape",
    "road",
    "busway",

    # Paths (non-car)
    "footway",
    "bridleway",
    "steps",
    "corridor",
    "path",
    "via_ferrata",
    "cycleway",

    # Other highway features (mostly nodes/POIs or non-street infrastructure)
    "bus_stop",
    "crossing",
    "cyclist_waiting_aid",
    "elevator",
    "emergency_access_point",
    "give_way",
    "hitchhiking",
    "ladder",
    "milestone",
    "mini_roundabout",
    "motorway_junction",
    "platform",
    "rest_area",
    "services",
    "speed_camera",
    "speed_display",
    "stop",
    "street_lamp",
    "toll_gantry",
    "traffic_mirror",
    "traffic_signals",
    "trailhead",
    "turning_circle",
    "turning_loop",
    "road",
    "proposed",
    "construction",
    "emergency_bay",
]