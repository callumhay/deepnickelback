import sys
import numpy as np
from pathlib import Path

import dnb_constants
import data_funcs

if len(sys.argv) < 3:
  print("Usage: python " + sys.argv[0] + " <spectrogram_dir> <spectrogram_file> <output_dir>")
  exit(-1)

spectrogramDir = sys.argv[1]
spectrogramPath = Path(spectrogramDir)
if not spectrogramPath.exists():
  print("Could not find directory " + spectrogramDir)
  exit(-1)
spectrogramPath = spectrogramPath.resolve()

specFile = sys.argv[2]
specFilePath = Path(specFile).resolve()

outputDir = sys.argv[3]
outputPath = Path(outputDir)
if not outputPath.exists():
  print("Could not find directory " + outputDir)
  exit(-1)
outputPath = outputPath.resolve()

outputFilePath = outputPath.joinpath(specFilePath.stem.replace("[NORMALIZED]", "[DENORMALIZED]") + ".npy").resolve()
sr = 22050

print("Loading file " + specFile + "...")
S = np.load(specFilePath)
S = data_funcs.denormalize(S, data_funcs.get_norm_spec_filepath(spectrogramPath))

np.save(outputFilePath, S)