

# Define constants
DEEP_NICKELBACK_VERSION = "0.9" # So far 0.8 is the best. 

NORMALIZED_DIR_NAME = "normalized"
NORMALIZED_SPEC_FILE_NAME = "norm_spec.txt"

# The number of mel channels across all files should be the same, get that number...
# This represents the number of frequency channels that the sound is split up into
# DO NOT CHANGE THIS NUMBER - IT DEFINES THE NUMBER OF INPUTS AND OUTPUTS OF THE RNN!!!
NUM_MEL_CHANNELS = 256
MEL_N_FFT = 2048