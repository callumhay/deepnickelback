

# Define constants
DEEP_NICKELBACK_VERSION = "0.21s" # So far 0.21s is best

NORMALIZED_DIR_NAME = "normalized"
NORMALIZED_SPEC_FILE_NAME = "norm_spec.txt"

STANDARDIZED_DIR_NAME = "standardized"
STANDARDIZED_SPEC_FILE_NAME_SUFFIX = "[standardized_spec].txt"

AVG_MEAN = 1.576460152854815
AVG_STD = 9.702951776040848


FULL_MODEL_FILE_NAME = "full_model.hdf5"

# The number of mel channels across all files should be the same, get that number...
# This represents the number of frequency channels that the sound is split up into
# DO NOT CHANGE THIS NUMBER - IT DEFINES THE NUMBER OF INPUTS AND OUTPUTS OF THE RNN!!!
NUM_MEL_CHANNELS = 256
MEL_N_FFT = 2048