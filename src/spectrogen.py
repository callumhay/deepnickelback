# First, load some audio and plot the spectrogram

import sys
import pickle
import matplotlib.pyplot as plt
import librosa
import soundfile

import numpy as np
import dnb_constants

from pathlib import Path


def save_unnormalized_spectrogram_file(currSoundFilePath, saveSpecFilePath):
  print("Loading file: " + str(currSoundFilePath))
  y, sr = librosa.load(currSoundFilePath)
  print("Generating Mel Spectrogram...")
  S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=dnb_constants.MEL_N_FFT, n_mels=dnb_constants.NUM_MEL_CHANNELS)
  print("Saving unnormalized Mel Spectrogram file: " + str(saveSpecFilePath))
  np.save(saveSpecFilePath, S)
  return S

if len(sys.argv) < 4:
  print("Usage: python " + sys.argv[0] + " <sound_file_dir> <spectrogram_dir> MODE=(global_normalize|standardize) POSTFIX=''")
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

mode = sys.argv[3]
soundFiles = list(soundFilePath.glob('*.wav'))+list(soundFilePath.glob('*.mp3'))

postfix = ""
if len(sys.argv) >= 5:
  postfix = sys.argv[4]

if mode == "global_normalize":

  min_val  = 0
  max_val  = 0

  for soundFile in soundFiles:
    spectrogramFilePath = spectrogramPath.joinpath(soundFile.stem + " [UNNORMALIZED].npy")
    if spectrogramFilePath.exists():
      S = np.load(spectrogramFilePath)
    else:
      S = save_unnormalized_spectrogram_file(soundFile.resolve(), spectrogramFilePath)
     
    # Figure out the min and the max...
    for i in range(len(S)):
      min_val = min(np.amin(S[i]), min_val)
      max_val = max(np.amax(S[i]), max_val)
    
  # Save the min and max to the normalized specfication file, 
  # this will help us do the reverse mapping at somepoint later on
  with open(spectrogramPath.joinpath(dnb_constants.NORMALIZED_DIR_NAME + "/" + dnb_constants.NORMALIZED_SPEC_FILE_NAME), 'w') as spec_file:
    spec_file.write(str(min_val) + "\n")
    spec_file.write(str(max_val) + "\n")

  # Now save out all the normalized data as well
  for soundFile in soundFiles:

    normSpectrogramFilePath = spectrogramPath.joinpath(dnb_constants.NORMALIZED_DIR_NAME + "/" + soundFile.stem + " [NORMALIZED]" + postfix + ".npy")
    if normSpectrogramFilePath.exists():
      continue

    currFilePath = str(soundFile.resolve())
    print("Loading file: " + currFilePath)
    y, sr = librosa.load(currFilePath)
    print("Generating Mel Spectrogram for audio in " + currFilePath)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=dnb_constants.MEL_N_FFT, n_mels=dnb_constants.NUM_MEL_CHANNELS)

    # Normalize the values in the S matrix
    for mel in range(len(S)):
      for i in range(len(S[mel])):
        S[mel][i] = (S[mel][i] - min_val) / (max_val - min_val)

    print("Saving normalized Mel Spectrogram file: " + str(normSpectrogramFilePath))
    np.save(normSpectrogramFilePath, S)

elif mode == "standardize":
  for soundFile in soundFiles:
    spectrogramFilePath = spectrogramPath.joinpath(soundFile.stem + " [UNNORMALIZED].npy")
    if spectrogramFilePath.exists():
      S = np.load(spectrogramFilePath)
    else:
      S = save_unnormalized_spectrogram_file(soundFile.resolve(), spectrogramFilePath)
    
    # Calculate the mean and standard deviation for the spectrogram then save them to file
    mean = np.mean(S)
    std  = np.std(S)

    with open(spectrogramPath.joinpath(dnb_constants.STANDARDIZED_DIR_NAME + "/" + soundFile.stem + postfix + " " + dnb_constants.STANDARDIZED_SPEC_FILE_NAME_SUFFIX), 'w') as spec_file:
      spec_file.write(str(mean) + "\n")
      spec_file.write(str(std) + "\n")

    stdSpectrogramFilePath = spectrogramPath.joinpath(dnb_constants.STANDARDIZED_DIR_NAME + "/" + soundFile.stem + postfix + " [STANDARDIZED].npy")
    if stdSpectrogramFilePath.exists():
      continue

    # Now generate the standardized spectrogram data file
    for mel in range(len(S)):
      for i in range(len(S[mel])):
        S[mel][i] = (S[mel][i] - mean) / std

    print("Saving standardized Mel Spectrogram file: " + str(stdSpectrogramFilePath))
    np.save(stdSpectrogramFilePath, S)

else:
  print("Could not find mode " + mode)