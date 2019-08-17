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

random.shuffle(spectroFiles) # shuffle the files for training

LOG_DIR = "./logs/training"
#file_writer = tf.contrib.summary.create_file_writer(LOG_DIR)
#tf.summary.trace_on(graph=True, profiler=True)

# Create and/or load our model with the most recent checkpoint
model = common.build_model(checkpoint_base_dir=checkpoint_base_dirpath)
model.summary()

'''
NUM_EPOCHS = 10000
NUM_SAVE_EPOCHS = 100
examples_per_epoch, batchedDataset = common.build_dataset_from_spectrograms(spectroFiles)
steps_per_epoch = examples_per_epoch//common.BATCH_SIZE 

checkpoint_prefix  = str(checkpoint_dirpath.joinpath("dnb_v" + str(dnb_constants.DEEP_NICKELBACK_VERSION) + "_epoch{epoch}").resolve())
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True, period=NUM_SAVE_EPOCHS, verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(LOG_DIR, histogram_freq=1, write_graph=True)
history = model.fit(batchedDataset.repeat(), epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback, tensorboard_callback])
'''

# Go through each of the spectrogram files in random order, we will create training data for each of them
file_count = 0
num_iter = 1000
for i in range(num_iter):
  for spectroFile in spectroFiles:
    resolvedPath = spectroFile.resolve()
    print("Loading file " + str(resolvedPath) + "...")

    # The format of S is a mel spectrogram, this is a 2D array with shape=(n_mels,t)
    # Convert the array to a set of per time-period training samples, 
    # this involves grouping the n_mels (frequency buckets) for every time segment together (i.e., transpose it)
    examples_per_epoch, batchedDataset = common.build_dataset_from_spectrogram(spectroFile)
    
    # NOTE: WE flip between two different checkpoint file names, saving over each one every 2nd epoch - this acts as
    # backup in case a save gets corrupted and also ensures we don't fill the hard drive with checkpoint files
    checkpoint_prefix   = str(checkpoint_dirpath.joinpath("dnb_v" + str(dnb_constants.DEEP_NICKELBACK_VERSION) + "_" + str(file_count % 2)).resolve())
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True, period=common.EPOCH_CHECKPOINT_SAVE_PERIOD, verbose=1)

    #tensorboard_callback = tf.keras.callbacks.TensorBoard(LOG_DIR, histogram_freq=1, write_graph=True)

    steps_per_epoch = examples_per_epoch//common.BATCH_SIZE 

    validation_dataset_size = steps_per_epoch//10 # Validation dataset should be about 10% of the total step size in each epoch
    validation_dataset = None
    train_dataset = None

    if validation_dataset_size > 0:
      validation_dataset = batchedDataset.take(validation_dataset_size).repeat()
      train_dataset = batchedDataset.skip(validation_dataset_size).repeat()
    else:
      validation_dataset_size = None
      train_dataset = batchedDataset.repeat()
    
    history = model.fit(train_dataset, validation_data=validation_dataset, validation_steps=validation_dataset_size, epochs=common.NUM_EPOCHS_PER_SONG, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

    model.reset_states()
    file_count += 1
