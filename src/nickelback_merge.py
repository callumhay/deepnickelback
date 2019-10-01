from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()

import librosa
import librosa.feature

import soundfile

import sys
import ast
import math
import argparse

import numpy as np
from pathlib import Path

import common
import dnb_constants
import data_funcs

parser = argparse.ArgumentParser(description='Merge an audio file with deep nickelback for a nickelbakin probably-not-so-great time.')
parser.add_argument('original_audio_filepath', metavar='audio_filepath', type=str, help='A file path to the original audio file or normalized numpy mel spectrogram you want to turn into a not-so-great one.')
parser.add_argument('spectrogram_norm_or_std_dirpath', metavar='spectrogram_norm_or_std_dirpath', type=str, help='A path to the directory where the normalization file for the spectrograms can be found.')
parser.add_argument('model_checkpoint_dirpath', metavar='model_checkpoint_dirpath', type=str, help='The path to the directory where the tensorflow model checkpoint is stored.')
parser.add_argument('--output_dirpath', metavar='output_dirpath', type=str, default=".", help='The output directory for the terrible result of this program.')
parser.add_argument('--duration', metavar='duration', type=int, default=None, help="The duration (in seconds) of the song that will be produced, if not provided then the entire song will be nickelbacked.")
args = parser.parse_args()
print(args)
# Attempt to load all the given file and directory paths...

original_audio_filepath = Path(args.original_audio_filepath)
if not original_audio_filepath.exists():
  print("Could not find audio file: " + str(original_audio_filepath))
  exit(-1)
original_audio_filepath = original_audio_filepath.resolve()

spectrogram_norm_or_std_dirpath = Path(args.spectrogram_norm_or_std_dirpath)
if not spectrogram_norm_or_std_dirpath.exists():
  print("Could not find spectrogram normalization/standardization directory: " + str(spectrogram_norm_or_std_dirpath))
  exit(-1)
spectrogram_norm_or_std_dirpath = spectrogram_norm_or_std_dirpath.resolve()

checkpt_base_dirpath = Path(args.model_checkpoint_dirpath)
if not checkpt_base_dirpath.exists():
  print("Could not find output directory: " + str(checkpt_base_dirpath))
  exit(-1)
checkpt_base_dirpath = checkpt_base_dirpath.resolve()

output_dirpath = Path(".")
if args.output_dirpath != None:
  output_dirpath = Path(args.output_dirpath)
if not output_dirpath.exists():
  print("Could not find output directory: " + str(output_dirpath))
  exit(-1)
output_dirpath = output_dirpath.resolve()

duration_in_s = args.duration
if duration_in_s != None and duration_in_s < 0:
  print("Invalid duration, must be an integer > 0")
  exit(-1)

# Restore the most recent deepnickelback model
print("Loading model...")

# Let's change the batch size of the model to be 1
model_version = dnb_constants.DEEP_NICKELBACK_VERSION
batch_size = 1

model = common.build_model(batch_size=batch_size, checkpoint_base_dir=checkpt_base_dirpath, version=model_version, stateful=True, compile=False)
model.build(tf.TensorShape([batch_size, None, dnb_constants.NUM_MEL_CHANNELS]))
model.summary()

S = None
if original_audio_filepath.suffix == ".npy":
  print("Loading spectrogram audio file...")
  S = np.load(original_audio_filepath)
else:
  print("Loading possible audio file...")

  y, sr = librosa.load(original_audio_filepath)
  print("Generating Mel Spectrogram...")
  S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=dnb_constants.MEL_N_FFT, n_mels=dnb_constants.NUM_MEL_CHANNELS)

  if "s" in model_version:
    # This is a standardized model so we standardize the spectrogram
    mean = np.mean(S)
    std = np.std(S)

    print("Standardizing Mel Spectrogram...")
    for mel in range(len(S)):
      for i in range(len(S[mel])):
        S[mel][i] = (S[mel][i] - mean) / std

  else:
    # Figure out the min and the max...
    min_val = 0
    max_val = 0
    for i in range(len(S)):
      max_val = max(np.amax(S[i]), max_val)

    print("Normalizing Mel Spectrogram...")
    for mel in range(len(S)):
      for i in range(len(S[mel])):
        S[mel][i] = (S[mel][i] - min_val) / (max_val - min_val)


print("Merging with the horribleness...")
sr = 22050
S = common.merge_terrible_mel_spectrogram(model, S, sr, duration_in_s, 2048)

# ... Now, de-normalize or de-standardize the spectrogram based on the values stored in the normalization spec file
if "s" in model_version:
  with open(data_funcs.get_std_spec_filepath(original_audio_filepath), 'r') as spec_file:
    mean = ast.literal_eval(spec_file.readline())
    std  = ast.literal_eval(spec_file.readline())

  mean = (mean + dnb_constants.AVG_MEAN) / 2.0
  std  = (std + dnb_constants.AVG_STD) / 2.0

  for mel in range(len(S)):
    for i in range(len(S[mel])):
      S[mel][i] = max(0, (S[mel][i] * std) + mean)

  out_spectrogram_filepath = output_dirpath.joinpath(original_audio_filepath.stem + "_spectrogram [DESTANDARDIZED].npy").resolve()
else:
  min_val = 0
  max_val = 0
  with open(spectrogram_norm_or_std_dirpath.joinpath(dnb_constants.NORMALIZED_SPEC_FILE_NAME), 'r') as spec_file:
    min_val = ast.literal_eval(spec_file.readline())
    max_val = ast.literal_eval(spec_file.readline())

  for mel in range(len(S)):
    for i in range(len(S[mel])):
      S[mel][i] = S[mel][i] * (max_val - min_val) + min_val

  out_spectrogram_filepath = output_dirpath.joinpath(original_audio_filepath.stem + "_spectrogram [DENORMALIZED].npy").resolve()


before_filepath = out_spectrogram_filepath
count = 1
while out_spectrogram_filepath.exists():
  out_spectrogram_filepath = before_filepath.with_name(before_filepath.stem + "(" + str(count) + ")" + before_filepath.suffix)
  count += 1

print("Saving resulting terrible mel spectrogram file: " + str(out_spectrogram_filepath))
np.save(out_spectrogram_filepath, S)