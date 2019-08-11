# First, load some audio and plot the spectrogram

import sys
import matplotlib.pyplot as plt
import librosa
import soundfile
import numpy as np

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

soundFiles = list(soundFilePath.glob('*.wav'))+list(soundFilePath.glob('*.mp3'))
print("The following sound files were found:")

for soundFile in soundFiles: 
  print(str(soundFile))

for soundFile in soundFiles:
  currFilePath = str(soundFile)
  print("Loading file " + currFilePath + "...")
  y, sr = librosa.load(currFilePath)
  print(y)
  
  if np.count_nonzero(y) < 1:
    print("Failed to properly load file, skipping...")
    continue

  print("Generating Mel Spectrogram...")
  S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=256)
  print(S)

  spectrogramFilePath = spectrogramPath.joinpath(soundFile.stem + ".npy")
  print("Saving Mel Spectrogram to file " + str(spectrogramFilePath))
  np.save(spectrogramFilePath, S)
  