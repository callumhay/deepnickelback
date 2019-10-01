
from pathlib import Path

import numpy as np
import ast
import dnb_constants

def get_norm_spec_filepath(spectrogram_dir=Path("./assets/spectrograms_256")):
  return spectrogram_dir.joinpath(dnb_constants.NORMALIZED_DIR_NAME + "/" + dnb_constants.NORMALIZED_SPEC_FILE_NAME).resolve()

def get_std_spec_filepath(original_filepath):
  return original_filepath.with_name(original_filepath.stem.replace(" [STANDARDIZED]", "") + " " + dnb_constants.STANDARDIZED_SPEC_FILE_NAME_SUFFIX).resolve()

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
  with open(norm_spec_filepath, 'r') as spec_file:
    min_val = ast.literal_eval(spec_file.readline())
    max_vals = ast.literal_eval(spec_file.readline())
  return denormalize_with_min_max(S, min_val, max_vals)


def destandardize_with_mean_std(S, mean, std):
  S_destd = np.copy(S)
  for mel in range(len(S_destd)):
    for i in range(len(S_destd[mel])):
      S_destd[mel][i] = max(0, (S_destd[mel][i] * std) + mean)

  return S_destd

def destandardize(S, std_spec_filepath):
  with open(std_spec_filepath, 'r') as spec_file:
    mean = ast.literal_eval(spec_file.readline())
    std  = ast.literal_eval(spec_file.readline())
  return destandardize_with_mean_std(S, mean, std)


def trim_zeros(S_t):
  start_idx = 0
  for i in range(len(S_t)):
    if np.count_nonzero(S_t[i]) > 0:
      break
    else:
      start_idx += 1

  end_idx = len(S_t)
  for i in range(len(S_t)-1,-1,-1):
    if np.count_nonzero(S_t[i]) > 0:
      break
    else:
      end_idx -= 1

  if start_idx >= end_idx:
    return []

  return S_t if start_idx == 0 and end_idx == len(S_t) else S_t[start_idx:end_idx]