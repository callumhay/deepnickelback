
import sys
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

from pathlib import Path

# Usage:
# spectogen.py <sound_file_dir> <spectogram_dir>

if len(sys.argv) < 2:
  print("Usage: python " + sys.argv[0] + " <spectrogram_numpy_file>")
  exit(-1)

specFile = sys.argv[1]
specFilePath = Path(specFile)

S = np.load(specFilePath)

print(len(S[0])) # the number of samples
print(len(S))    # number of mel bands

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram (' + specFilePath.name + ')')
plt.tight_layout()
plt.show()