from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()

import librosa
import librosa.feature

import soundfile

import sys
import ast
import math
import random

import numpy as np
from pathlib import Path

import common
import dnb_constants

if len(sys.argv) < 6:
  print("Usage: python " + sys.argv[0] + " <seed_spectrogram_filepath> <output_dir> <song_duration_in_secs> <checkpoint_base_dir> <spectrogram_path>")
  exit(-1)

seed_file_str = sys.argv[1]
seed_filepath = Path(seed_file_str)
if not seed_filepath.exists():
  print("Could not find seed music file: " + seed_file_str)
  exit(-1)
seed_filepath = seed_filepath.resolve()

output_dir_str = sys.argv[2]
output_dirpath = Path(output_dir_str)
if not output_dirpath.exists():
  print("Could not find output directory: " + output_dir_str)
  exit(-1)
output_dirpath = output_dirpath.resolve()

song_duration_s = int(sys.argv[3])
if song_duration_s <= 0:
  print("Song length must be a positive integer greater than zero.")
  exit(-1)

checkpt_base_dir_str = sys.argv[4]
checkpt_base_dirpath = Path(checkpt_base_dir_str)
if not checkpt_base_dirpath.exists():
  print("Could not find output directory: " + checkpt_base_dir_str)
  exit(-1)
checkpt_base_dirpath = checkpt_base_dirpath.resolve()

spectrogram_dir_str = sys.argv[5]
spectrogram_dirpath = Path(spectrogram_dir_str)
if not spectrogram_dirpath.exists():
  print("Could not find spectrogram directory: " + spectrogram_dir_str)
  exit(-1)
spectrogram_dirpath = spectrogram_dirpath.resolve()

# Generate a very brief spectrogram from the given seed music file...
seed_filepath_str = str(seed_filepath)
print("Loading seed file " + seed_filepath_str + "...")
seed_S = np.load(seed_file_str)

# Now the fun part: Use the seed data to feed the first part of the RNN and let the model feedback into itself
# to generate enough data to make a song of the specified song length

# Restore the most recent deepnickelback model
print("Loading model...")
# Let's change the batch size of the model to be just 1
model = common.build_model(batch_size=1, checkpoint_base_dir=checkpt_base_dirpath)
model.summary()

print("Generating terrible music...")
sr = 22050
S = common.generate_terrible_mel_spectrogram(model, seed_S, sr, song_duration_s)

# Save out the normalized spectrogram first
#spectrogram_filepath = output_dirpath.joinpath(seed_filepath.stem + "_spectrogram [NORMALIZED].npy").resolve()
#print("Saving intermediate mel spectrogram file: " + str(spectrogram_filepath))
#np.save(spectrogram_filepath, S)

# ... Now, de-normalize the spectrogram based on the values stored in the normalization spec file
min_val = 0
max_vals = []
with open(spectrogram_dirpath.joinpath(dnb_constants.NORMALIZED_DIR_NAME + "/" + dnb_constants.NORMALIZED_SPEC_FILE_NAME), 'r') as spec_file:
  min_val = ast.literal_eval(spec_file.readline())
  max_vals = ast.literal_eval(spec_file.readline())

for mel in range(len(S)):
  for i in range(len(S[mel])):
    S[mel][i] = S[mel][i] * (max_vals[mel] - min_val) + min_val

# ... and save out the de-normalized spectrogram as well
spectrogram_filepath = output_dirpath.joinpath(seed_filepath.stem + "_spectrogram [DENORMALIZED].npy").resolve()
print("Saving normalized mel spectrogram file: " + str(spectrogram_filepath))
np.save(spectrogram_filepath, S)

'''
print("Converting mel spectrogram to audio...")
y = librosa.feature.inverse.mel_to_audio(S, sr)

output_filepath_str = str(output_dirpath.joinpath(seed_filepath.stem + "_made_terrible.wav").resolve())
print("Writing terrible audio file: " + output_filepath_str + "...")
soundfile.write(output_filepath_str, y, sr)
print("DONE - Hope you don't regret this.")
'''