import sys
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile
import numpy as np

from pathlib import Path

if len(sys.argv) < 3:
  print("Usage: python " + sys.argv[0] + " <spectrogram_file> <output_dir>")
  exit(-1)

specFile = sys.argv[1]
specFilePath = Path(specFile).resolve()

outputDir = sys.argv[2]
outputPath = Path(outputDir)
if not outputPath.exists():
  print("Could not find directory " + outputDir)
  exit(-1)

audioFilePath = outputPath.joinpath(specFilePath.stem + " output" + '.wav').resolve()
sr = 22050

print("Loading file " + specFile + "...")
S = np.load(specFilePath)
print(S)

print("Converting mel spectrogram to audio...")
y = librosa.feature.inverse.mel_to_audio(S, sr)

print("Writing audio file " + str(audioFilePath) + "...")
soundfile.write(str(audioFilePath), y, sr)