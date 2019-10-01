from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()

import sys
import random
import math

import numpy as np
from pathlib import Path

import common
import dnb_constants

if len(sys.argv) < 3:
  print("Usage: python " + sys.argv[0] + " <spectrogram_dir> <checkpoint_base_dir> [num_epochs=100]")
  exit(-1)

spectroDir = sys.argv[1]
spectroDirPath = Path(spectroDir)
if not spectroDirPath.exists():
  print("Could not find spectrogram directory: " + spectroDir)
  exit(-1)

checkpoint_base_dir = sys.argv[2]
checkpoint_base_dirpath = Path(checkpoint_base_dir)
if not checkpoint_base_dirpath.exists():
  print("Could not find checkpoint directory: " + checkpoint_base_dir)
  exit(-1)
checkpoint_dirpath = common.get_latest_checkpoint_dir(checkpoint_base_dirpath)

# Generate the deepnickelback network from a given directory of 
# mel spectrographs of each of their terrible songs
spectroFiles = list(spectroDirPath.glob('*.npy'))
print("The following spectrogram files were found:")
for sf in spectroFiles: 
  print(str(sf))

#LOG_DIR = "./logs/training"
#file_writer = tf.contrib.summary.create_file_writer(LOG_DIR)
#tf.summary.trace_on(graph=True, profiler=True)

# Create and/or load our model with the most recent checkpoint
CURR_BATCH_SIZE = common.BATCH_SIZE
model = common.build_model(batch_size=CURR_BATCH_SIZE, checkpoint_base_dir=checkpoint_base_dirpath, stateful=False)
model.summary()
model.reset_states()

checkpoint_prefix   = str(checkpoint_dirpath.joinpath("dnb_v" + dnb_constants.DEEP_NICKELBACK_VERSION + "_epoch{epoch:04d}_loss{loss:.8f}")) #str(checkpoint_dirpath.joinpath(dnb_constants.FULL_MODEL_FILE_NAME).resolve())
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True, monitor='loss', save_best_only=True, period=1, verbose=1)
scheduler_callback  = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', mode='min', factor=0.75, patience=5, min_lr=1e-10, min_delta=1e-6, verbose=1)#tf.keras.callbacks.LearningRateScheduler(common.learning_scheduler)
#tensorboard_callback = tf.keras.callbacks.TensorBoard(LOG_DIR, histogram_freq=1, write_graph=True)

steps_per_epoch = common.generator_steps_per_epoch(spectroFiles)
history = model.fit_generator(common.generator_dataset_func_from_spectrograms(spectroFiles, CURR_BATCH_SIZE), epochs=500, steps_per_epoch=steps_per_epoch, shuffle=False, callbacks=[checkpoint_callback, scheduler_callback])
