from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()

import sys
import random
import numpy as np

import common

from pathlib import Path

if len(sys.argv) < 3:
  print("Usage: python " + sys.argv[0] + " <spectrogram_dir> <checkpoint_dir>")
  exit(-1)

spectroDir = sys.argv[1]
spectroDirPath = Path(spectroDir)
if not spectroDirPath.exists():
  print("Could not find directory " + spectroDir)
  exit(-1)

checkpointDir = sys.argv[2]
checkpointDirPath = Path(checkpointDir)
if not checkpointDirPath.exists():
  print("Could not find directory " + spectroDir)
  exit(-1)

model = common.build_model()
model.summary()

spectroFiles = list(spectroDirPath.glob('*.npy'))
random.shuffle(spectroFiles)
S, batchedDataset = common.build_dataset_from_spectrogram(spectroFiles[0].resolve())

loss, acc = model.evaluate(batchedDataset.repeat(), steps=100)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

model.load_weights(tf.train.latest_checkpoint(str(common.get_latest_checkpoint_dir(checkpointDirPath))))
loss, acc = model.evaluate(batchedDataset.repeat(), steps=100)
print("Trained model, accuracy: {:5.2f}%".format(100*acc))
