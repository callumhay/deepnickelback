import sys
import numpy as np
from pathlib import Path

import dnb_constants
import data_funcs

if len(sys.argv) < 4:
  print("Usage: python " + sys.argv[0] + " <spectrogram_dir> <spectrogram_file> <output_dir> <(denormalize|destandardize)>")
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

process_type = sys.argv[4]

if process_type == "denormalize":
  outputFilePath = outputPath.joinpath(specFilePath.stem.replace("[NORMALIZED]", "[DENORMALIZED]") + ".npy").resolve()
  sr = 22050
  print("Loading file " + specFile + "...")
  S = np.load(specFilePath)
  S = data_funcs.denormalize(S, data_funcs.get_norm_spec_filepath(spectrogramPath))

elif process_type == "destandardize":
  outputFilePath = outputPath.joinpath(specFilePath.stem.replace("[STANDARDIZED]", "[DESTANDARDIZED]") + ".npy").resolve()
  sr = 22050
  print("Loading file " + specFile + "...")
  S = np.load(specFilePath)
  S = data_funcs.destandardize(S, data_funcs.get_std_spec_filepath(specFilePath))

else:
  print("Process type '" + process_type + "' not found.")
  exit(-1)

print("Saving file " + str(outputFilePath))
np.save(outputFilePath, S)