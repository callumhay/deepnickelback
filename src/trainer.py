from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()

import sys
import numpy as np

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


# Generate the deepnickelback network from a given directory of 
# mel spectrographs of each of their terrible songs

spectroFiles = list(spectroDirPath.glob('*.npy'))
print("The following spectrogram files were found:")
for sf in spectroFiles: 
  print(str(sf))

# Define constants
DEEP_NICKELBACK_VERSION = "0.1"
SEQ_LENGTH = 10*10         # Since the samples are approx 1024ms each this represents about 0.1*SEQ_LENGTH seconds of consective sampled song data
BATCH_SIZE = 6             # Represents SEQ_LENGTH*x seconds of sampled song data
NUM_RNN_0_UNITS = 1024     # Number of RNN nodes/units in the 0th layer of the network
NUM_EPOCHS_PER_SONG = 100  # Number of epochs to run training on the model for each song

def build_model(batch_size, input_size, output_size, num_rnn_units):
  return tf.keras.Sequential([
    tf.keras.layers.CuDNNLSTM(num_rnn_units, return_sequences=True, stateful=True, batch_input_shape=(batch_size, None, input_size)),
    tf.keras.layers.Dense(output_size)
  ])

def loss(targets, predictions):
  return tf.keras.losses.MSE(tf.cast(targets, dtype=tf.float32), tf.cast(predictions, dtype=tf.float32))

# Go through each of the spectrogram files, we will create training data for each of them
# Training data will need a sequence length (i.e., the number of audio samples per training dataset)
num_mel_channels = None
model = None
for spectroFile in spectroFiles:
  resolvedPath = spectroFile.resolve()
  print("Loading file " + str(resolvedPath) + "...")

  # The format of S is a mel spectrogram, this is a 2D array with shape=(n_mels,t)
  # Convert the array to a set of per time-period training samples, 
  # this involves grouping the n_mels (frequency buckets) for every time segment together (i.e., transpose it)
  S = np.transpose(np.load(spectroFile)) 

  s_dataset = tf.data.Dataset.from_tensor_slices(S)

  # The number of mel channels across all files should be the same, get that number...
  # This represents the number of frequency channels that the sound is split up into
  num_mel_channels = len(S[0])

  # Build the model if it doesn't exist yet
  if model is None:
    model = build_model(BATCH_SIZE, num_mel_channels, num_mel_channels, NUM_RNN_0_UNITS)
    model.summary()
    model.compile(tf.train.AdamOptimizer(), loss=loss)



'''
print(s_dataset.output_shapes)
for x in s_dataset: 
  print(x)
'''

# Break the dataset up into batched sequences for training
sequences = s_dataset.batch(SEQ_LENGTH+1, drop_remainder=True)

def split_input_target(chunk):
  input_samples = chunk[:-1]
  target_samples = chunk[1:]
  return input_samples, target_samples

dataset = sequences.map(split_input_target)
'''
print(dataset)  #shape: (SEQ_LENGTH, n_mels)
for input_example, target_example in dataset.take(1):
  #print ('Input data: ' + str(input_example[1]))
  #print ('Target data:' + str(target_example[0]))

  for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(input_idx)))
    print("  expected output: {} ({:s})".format(target_idx, repr(target_idx)))
'''

batchedDataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
#print(batchedDataset) #shape: (BATCH_SIZE, SEQ_LENGTH, n_mels)


checkpoint_prefix   = str(checkpointDirPath.joinpath("dnb_v" + str(DEEP_NICKELBACK_VERSION) + "_{epoch}").resolve())
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

examples_per_epoch = len(S)//SEQ_LENGTH
steps_per_epoch = examples_per_epoch//BATCH_SIZE 
history = model.fit(batchedDataset.repeat(), epochs=NUM_EPOCHS_PER_SONG, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])


'''
for input_example_batch, target_example_batch in batchedDataset.take(1):
  f32_input_example_batch  = tf.cast(input_example_batch, dtype=tf.float32)
  f32_target_example_batch = tf.cast(target_example_batch, dtype=tf.float32)

  example_batch_predictions = model(f32_input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
  
  example_batch_loss = loss(f32_input_example_batch, f32_target_example_batch)
  print("scalar_loss: ", example_batch_loss.numpy().mean())
'''