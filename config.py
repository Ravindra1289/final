import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
DATASET_DIR = os.path.join(OUTPUT_DIR, "datasets")

for dir_path in [MODEL_DIR, DATASET_DIR, os.path.join(OUTPUT_DIR, "plots"), os.path.join(OUTPUT_DIR, "logs")]:
    os.makedirs(dir_path, exist_ok=True)

# Grid Settings
DEFAULT_MAP_SIZE = 20
DEFAULT_OBSTACLE_DENSITY = 0.2

# Kinematics & Orientations
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
FORWARD, LEFT, RIGHT = 0, 1, 2

MOVES = {
    NORTH: (-1, 0),
    EAST: (0, 1),
    SOUTH: (1, 0),
    WEST: (0, -1)
}

# Energy Weights
DISTANCE_COST = 1.0
TURN_COST = 5.0
SLOPE_COST = 10.0
MAX_SLOPE = 0.5  # If height_next - height_current > MAX_SLOPE, treat as obstacle

# Training Defaults
NUM_MAPS_TRAIN = 500
SCENARIOS_PER_MAP = 50
