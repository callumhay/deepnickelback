# First, load some audio and plot the spectrogram

import sys
import pickle
import matplotlib.pyplot as plt
import librosa
import soundfile

import numpy as np
import dnb_constants

from pathlib import Path

if len(sys.argv) < 3:
  print("Usage: python " + sys.argv[0] + " <sound_file_dir> <spectrogram_dir>")
  exit(-1)

soundFileDir = sys.argv[1]
soundFilePath = Path(soundFileDir)
if not soundFilePath.exists():
  print("Could not find directory " + soundFileDir)
  exit(-1)

spectrogramDir = sys.argv[2]
spectrogramPath = Path(spectrogramDir)
if not spectrogramPath.exists():
  print("Could not find directory " + spectrogramDir)
  exit(-1)

min_val  = 0
max_vals = np.zeros(dnb_constants.NUM_MEL_CHANNELS)
soundFiles = list(soundFilePath.glob('*.wav'))+list(soundFilePath.glob('*.mp3'))

for soundFile in soundFiles:
  currFilePath = str(soundFile.resolve())

  print("Loading file: " + currFilePath)
  y, sr = librosa.load(currFilePath)

  print("Generating Mel Spectrogram for audio in " + currFilePath)
  S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=dnb_constants.MEL_N_FFT, n_mels=dnb_constants.NUM_MEL_CHANNELS)

  spectrogramFilePath = spectrogramPath.joinpath(soundFile.stem + " [UNNORMALIZED].npy")
  print("Saving unnormalized Mel Spectrogram file: " + str(spectrogramFilePath))
  np.save(spectrogramFilePath, S)

  # Figure out the min and the max...
  for i in range(len(S)):
    max_vals[i] = max(np.amax(S[i]), max_vals[i])

# Save the min and max to the normalized specfication file, 
# this will help us do the reverse mapping at somepoint later on
with open(spectrogramPath.joinpath(dnb_constants.NORMALIZED_DIR_NAME + "/" + dnb_constants.NORMALIZED_SPEC_FILE_NAME), 'w') as spec_file:
  spec_file.write(str(min_val) + "\n")
  spec_file.write(str(max_vals).replace("\n", "") + "\n")

# Now save out all the normalized data as well
for soundFile in soundFiles:

  currFilePath = str(soundFile.resolve())
  print("Loading file: " + currFilePath)
  y, sr = librosa.load(currFilePath)
  print("Generating Mel Spectrogram for audio in " + currFilePath)
  S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=dnb_constants.MEL_N_FFT, n_mels=dnb_constants.NUM_MEL_CHANNELS)

  for mel in range(len(S)):
    for i in range(len(S[mel])):
      S[mel][i] = (S[mel][i] - min_val) / (max_vals[mel] - min_val)

  # Normalize the values in the S matrix
  normSpectrogramFilePath = spectrogramPath.joinpath(dnb_constants.NORMALIZED_DIR_NAME + "/" + soundFile.stem + " [NORMALIZED].npy")
  print("Saving normalized Mel Spectrogram file: " + str(normSpectrogramFilePath))
  np.save(normSpectrogramFilePath, S)
