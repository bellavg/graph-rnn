# G2PT/configs/aig.py
import math
import os # Added for potential path joining if needed


# --- Primary Configuration Constants ---
dataset = 'aig'
block_size = 768
vocab_size = 94
# Consider using os.path.join for better path handling
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Gets G2PT directory
# data_dir = os.path.join(base_dir, 'datasets', 'aig')
# tokenizer_path = os.path.join(base_dir, 'tokenizers', 'aig')
# FOR G2PT only
data_dir = '.G2PT/datasets/aig/' # Keeping relative path as in original
tokenizer_path = '.G2PT/datasets/aig/tokenizer' # Keeping relative path as in original


# Padding value for loss calculation (ignore_index)
PAD_VALUE = -100

# AIG constraint constants
MAX_NODE_COUNT = 64
MIN_PI_COUNT = 2
MAX_PI_COUNT = 8
MIN_PO_COUNT = 1
MAX_PO_COUNT = 8
MIN_AND_COUNT = 1 # Assuming at least one AND gate needed


# Define the base token types in order
# *** Renamed to Uppercase Constants ***
STRUCTURE_TOKENS = ["<boc>", "<eoc>", "<sepc>", "<bog>", "<eog>", "<sepg>"]
NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]

# --- Derived Vocabulary Generation ---

# Generate index tokens based on MAX_NODE_COUNT
# *** Renamed to Uppercase Constant ***
IDX_TOKENS = [f"IDX_{i}" for i in range(MAX_NODE_COUNT)] # IDX_0 to MAX_NODE_COUNT-1

# Generate PI count tokens based on MIN/MAX PI counts
# *** Renamed to Uppercase Constant ***
PI_COUNT_TOKENS = [f"PI_COUNT_{i}" for i in range(MIN_PI_COUNT, MAX_PI_COUNT + 1)]

# Generate PO count tokens based on MIN/MAX PO counts
# *** Renamed to Uppercase Constant ***
PO_COUNT_TOKENS = [f"PO_COUNT_{i}" for i in range(MIN_PO_COUNT, MAX_PO_COUNT + 1)]

# Combine all tokens for the main vocabulary in the desired order
# *** Renamed to Uppercase Constant ***
ORDERED_MAIN_TOKENS = (
    STRUCTURE_TOKENS +
    IDX_TOKENS +
    NODE_TYPE_KEYS +
    EDGE_TYPE_KEYS +
    PI_COUNT_TOKENS +
    PO_COUNT_TOKENS
)

# Create the main vocabulary dictionary with sequential integer values (0, 1, 2, ...)
# Example: {"<boc>": 0, "<eoc>": 1, ..., "IDX_0": 6, ..., "PO_COUNT_8": 90}
FULL_VOCAB = {token: i for i, token in enumerate(ORDERED_MAIN_TOKENS)}

# Define special tokenizer tokens (like UNK, PAD, MASK) and their IDs
# Ensure these IDs are consistent with your tokenizer's configuration file (e.g., tokenizer.json)
SPECIAL_TOKENS = {
    "[UNK]": 91,
    "[PAD]": 92,
    "[MASK]": 93
}

# Define constants for easy access to special token strings
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
# Keep PAD_TOKEN_ID derived for potential use elsewhere or specific checks.
PAD_TOKEN_ID = SPECIAL_TOKENS[PAD_TOKEN]
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS) # Dynamically count defined special tokens


# --- Derive other vocabulary-related constants ---

# Extract node/edge specific vocabularies from FULL_VOCAB based on the defined keys
# Example: {"NODE_CONST0": 70, "NODE_PI": 71, ...}
NODE_TYPE_VOCAB = {k: FULL_VOCAB[k] for k in NODE_TYPE_KEYS if k in FULL_VOCAB}
# Example: {"EDGE_REG": 74, "EDGE_INV": 75}
EDGE_TYPE_VOCAB = {k: FULL_VOCAB[k] for k in EDGE_TYPE_KEYS if k in FULL_VOCAB}

# Derive offsets (minimum value for node/edge types in the FULL_VOCAB)
NODE_VOCAB_OFFSET = min(NODE_TYPE_VOCAB.values()) if NODE_TYPE_VOCAB else -1 # Should be 70
EDGE_VOCAB_OFFSET = min(EDGE_TYPE_VOCAB.values()) if EDGE_TYPE_VOCAB else -1 # Should be 74

# Derive feature counts from the size of the derived vocabularies
NUM_NODE_FEATURES = len(NODE_TYPE_VOCAB) # Should be 4
NUM_EDGE_FEATURES = len(EDGE_TYPE_VOCAB) # Should be 2

# VALID_AIG_NODE_TYPES and VALID_AIG_EDGE_TYPES removed - derive when needed using set(NODE_TYPE_VOCAB.keys()) etc.

# Derive feature index to vocab mapping (assuming order matches NODE_TYPE_KEYS/EDGE_TYPE_KEYS)
# This maps a feature index (0, 1, 2...) to the corresponding token ID from FULL_VOCAB
# Example: {0: 70, 1: 71, 2: 72, 3: 73}
NODE_FEATURE_INDEX_TO_VOCAB = {i: NODE_TYPE_VOCAB[k] for i, k in enumerate(NODE_TYPE_KEYS)}
# Example: {0: 74, 1: 75}
EDGE_FEATURE_INDEX_TO_VOCAB = {i: EDGE_TYPE_VOCAB[k] for i, k in enumerate(EDGE_TYPE_KEYS)}

# Define one-hot encodings (Keep these hardcoded as they define the feature representation)
NODE_TYPE_ENCODING = {
    # Map "NODE_CONST0" to the first encoding vector (index 0), etc.
    "NODE_CONST0": [1.0, 0.0, 0.0, 0.0], # Index 0 feature
    "NODE_PI":     [0.0, 1.0, 0.0, 0.0], # Index 1 feature
    "NODE_AND":    [0.0, 0.0, 1.0, 0.0], # Index 2 feature
    "NODE_PO":     [0.0, 0.0, 0.0, 1.0]  # Index 3 feature
}

# IMPORTANT: Ensure keys/order match EDGE_TYPE_VOCAB derivation and NUM_EDGE_FEATURES
# The order here should ideally match the order in EDGE_TYPE_KEYS
EDGE_LABEL_ENCODING = {
    "EDGE_REG": [1.0, 0.0],  # Index 0 feature
    "EDGE_INV": [0.0, 1.0]   # Index 1 feature
}

# --- Final Vocab Size Calculation ---
# Determine the highest ID used across both the main vocab and special tokens
# *** Renamed internal calculation vars to Uppercase Constants ***
MAX_FULL_VOCAB_ID = max(FULL_VOCAB.values()) if FULL_VOCAB else -1 # Should be 90
MAX_SPECIAL_ID = max(SPECIAL_TOKENS.values()) if SPECIAL_TOKENS else -1 # Should be 93
OVERALL_MAX_ID = max(MAX_FULL_VOCAB_ID, MAX_SPECIAL_ID) # Should be 93

# vocab_size is typically the number of unique tokens = highest ID + 1


# --- Assertions Removed ---
# The import and call to check_aig_config have been removed.
# You can now import this config file without automatically running checks.
# To run checks, import check_aig_config from .aig_assertions
# and call it explicitly in a separate script, passing the config values.
# Example (in another file like tests/test_config.py):
#
# import G2PT.configs.aig as aig_config
# from G2PT.configs.aig_assertions import check_aig_config
#
# check_aig_config() # If using 'from .aig import *' in assertions file
# print("Config checks passed manually.")

