
from pathlib import Path

import numpy as np
import ast
import dnb_constants

def get_norm_spec_filepath(spectrogram_dir=Path("./assets/spectrograms_256")):
  return spectrogram_dir.joinpath(dnb_constants.NORMALIZED_DIR_NAME + "/" + dnb_constants.NORMALIZED_SPEC_FILE_NAME).resolve()

def normalize_with_min_max(S, min_val, max_vals):
  assert(len(S) == len(max_vals))
  S_norm = np.copy(S)
  for mel in range(len(S_norm)):
    for i in range(len(S_norm[mel])):
      S_norm[mel][i] = (S_norm[mel][i] - min_val) / (max_vals[mel] - min_val)

  return S_norm

def normalize(S, norm_spec_filepath):
  min_val = 0
  max_vals = []
  with open(norm_spec_filepath, 'r') as spec_file:
    min_val = ast.literal_eval(spec_file.readline())
    max_vals = ast.literal_eval(spec_file.readline())

  return normalize_with_min_max(S, min_val, max_vals)


def denormalize_with_min_max(S, min_val, max_vals):
  S_denorm = np.copy(S)
  for mel in range(len(S_denorm)):
    for i in range(len(S_denorm[mel])):
      S_denorm[mel][i] = S_denorm[mel][i] * (max_vals[mel] - min_val) + min_val

  return S_denorm

##
# Denormalizes the given numpy array based on the spectrogram normalization spec file, located at the given directory.
##
def denormalize(S, norm_spec_filepath):
  min_val = 0
  max_vals = []
  with open(norm_spec_filepath, 'r') as spec_file:
    min_val = ast.literal_eval(spec_file.readline())
    max_vals = ast.literal_eval(spec_file.readline())

  return denormalize_with_min_max(S, min_val, max_vals)