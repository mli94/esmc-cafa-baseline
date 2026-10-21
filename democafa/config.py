import yaml
import os

DATA_DIR = os.environ.get('DATA_DIR', 'data')
UTILS_DIR = os.environ.get('UTILS_DIR', 'democafa')

def load_config(config_file="config.yaml"):
    """Loads configuration from a YAML file."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(current_dir, config_file)
    with open(config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"Error loading config file: {e}")
            return None

config = load_config() # Load configuration at module import time

if config:
    VERSIONS = config.get('versions', {}) 
    GO_CODES = config.get('go_codes', {}) 
    RAW_FILE_PATHS = config.get('raw_file_paths', {})
    for key in RAW_FILE_PATHS:
        RAW_FILE_PATHS[key] = os.path.join(DATA_DIR, RAW_FILE_PATHS[key])
    PROCESSED_FILE_PATHS = config.get('processed_file_paths', {})
    for key in PROCESSED_FILE_PATHS:
        PROCESSED_FILE_PATHS[key] = os.path.join(DATA_DIR, PROCESSED_FILE_PATHS[key])
    RELEASE_PATHS = config.get('release_paths', {})
    for key in RELEASE_PATHS:
        RELEASE_PATHS[key] = os.path.join(DATA_DIR, RELEASE_PATHS[key])
    EXTERNAL_TOOLS = config.get('external_tools', {})
    for key in EXTERNAL_TOOLS:
        EXTERNAL_TOOLS[key] = os.path.join(UTILS_DIR, EXTERNAL_TOOLS[key])
    BASELINE_PATHS = config.get('baseline_paths', {})
    for key in BASELINE_PATHS:
        BASELINE_PATHS[key] = os.path.join(DATA_DIR, BASELINE_PATHS[key])
    GROUNDTRUTH_PATHS = config.get('groundtruth_paths', {})
    for key in GROUNDTRUTH_PATHS:
        GROUNDTRUTH_PATHS[key] = os.path.join(DATA_DIR, GROUNDTRUTH_PATHS[key])
else:
    VERSIONS = {}
    GO_CODES = {}
    RAW_FILE_PATHS = {}
    PROCESSED_FILE_PATHS = {}
    RELEASE_PATHS = {}
    GROUNDTRUTH_PATHS = {}
