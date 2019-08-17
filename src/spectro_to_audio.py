import sys
import matplotlib.pyplot as plt
import librosa
import librosa.feature
import soundfile
import numpy as np
import math
import time

from pathlib import Path

import dnb_constants


def convert_spectrogram(specFilePath, outputPath):
  audioFilePath = outputPath.joinpath(specFilePath.stem + " output.wav").resolve()
  sr = 22050

  print("Loading file " + specFile + "...")
  S = np.load(specFilePath)
  if len(sys.argv) > 3:
    number_of_samples = int(sys.argv[3])
    S = np.delete(S, range(number_of_samples,len(S[0])), 1)

  print("Converting mel spectrogram to audio...")
  y = librosa.feature.inverse.mel_to_audio(S, sr, n_fft=dnb_constants.MEL_N_FFT)

  print("Writing audio file " + str(audioFilePath) + "...")
  soundfile.write(str(audioFilePath), y, sr)



if len(sys.argv) < 3:
  print("Usage: python " + sys.argv[0] + " <spectrogram_file_or_dir> <output_dir> <number_of_samples=full_duration>")
  exit(-1)


specFile = sys.argv[1]
specFilePath = Path(specFile)
if not specFilePath.exists():
  print("Could not find file/directory: " + specFile)
  exit(-1)
specFilePath = specFilePath.resolve()

outputDir = sys.argv[2]
outputPath = Path(outputDir)
if not outputPath.exists():
  print("Could not find directory: " + outputDir)
  exit(-1)
outputPath = outputPath.resolve()

# If we were given a directory to look in for spectrograms, then we watch for files in that directory
# convert them as they appear/change

if specFilePath.is_dir():

  specFileDir = specFilePath
  converted_files = {}

  while True:
    print ("Listening on directory: " + str(specFileDir) + "...")
    # Grab all the spectrogram files...
    found_files = list(specFileDir.glob('*.npy'))

    # Check if we've seen any before or if they're new
    new_and_changed_files = []
    for f in found_files:
      file_str = str(f.resolve())

      if file_str in converted_files:
        # Check the file modification timestamp
        if converted_files[file_str].stat().st_mtime < f.stat().st_mtime:
          new_and_changed_files.append(f.resolve())
      else:
        # New file found
        new_and_changed_files.append(f.resolve())

    # Clean-up any converted file entries where the original has been deleted
    deleted_files = []
    for f_str in converted_files:
      if not converted_files[f_str].exists():
        deleted_files.append(f_str)
    for f_str in deleted_files:
      del converted_files[f_str]

    # Convert all the new and changed files
    for f in new_and_changed_files:
      convert_spectrogram(f, outputPath)
      converted_files[str(f)] = f

    time.sleep(5)

else:
  # We were just given a single file to convert
  convert_spectrogram(specFilePath, outputPath)