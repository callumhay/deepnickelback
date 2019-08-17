import tensorflow as tf
import numpy as np
from pathlib import Path

import math

import dnb_constants

NUM_RNN_0_UNITS = 1024   # Number of RNN nodes/units in the nth layer of the network
NUM_RNN_1_UNITS = 512
NUM_RNN_2_UNITS = 256
NUM_RNN_3_UNITS = 256

SEQ_LENGTH  = 16   # Each sample in the sequence represents approx 22050/dnb_constants.MEL_N_FFT = 0.09287982 seconds of total time in a song
BATCH_SIZE  = 32   # Represents SEQ_LENGTH*BATCH_SIZE seconds of sampled song data
BUFFER_SIZE = 10000

NUM_EPOCHS_PER_SONG = 100  # Number of epochs to run training on the model for each song
EPOCH_CHECKPOINT_SAVE_PERIOD = NUM_EPOCHS_PER_SONG

##
# Gets the string of the latest checkpoint filepath.
##
def get_latest_checkpoint_dir(base_dirpath, version=dnb_constants.DEEP_NICKELBACK_VERSION):
  return base_dirpath.joinpath(version).resolve()

##
# Builds the RNN model for tensorflow. If the checkpoint directory (python pathlib object) is given then it will
# load the most recent checkpoint from it for the given version of the model.
##
def build_model(batch_size=BATCH_SIZE, version=dnb_constants.DEEP_NICKELBACK_VERSION, checkpoint_base_dir=None):
  model = None
  optimizer = None
  if version == "0.1": 
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(NUM_RNN_0_UNITS, return_sequences=True, stateful=True, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS)
    ])
    optimizer = tf.train.AdamOptimizer()
  elif version == "0.2":
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(NUM_RNN_0_UNITS, return_sequences=True, stateful=True, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.CuDNNLSTM(NUM_RNN_1_UNITS, return_sequences=True, stateful=True),
      tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS)
    ])
    optimizer = tf.train.AdamOptimizer()
  elif version == "0.3":
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(NUM_RNN_0_UNITS, return_sequences=True, stateful=True, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.Dropout(rate=0.25),
      tf.keras.layers.CuDNNLSTM(NUM_RNN_1_UNITS, return_sequences=True, stateful=True),
      tf.keras.layers.Dropout(rate=0.25),
      tf.keras.layers.CuDNNLSTM(NUM_RNN_2_UNITS, return_sequences=True, stateful=True),
      tf.keras.layers.Dropout(rate=0.25),
      tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS)
    ])
    optimizer = tf.train.AdamOptimizer()
  elif version == "0.4":
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(NUM_RNN_0_UNITS, return_sequences=True, stateful=True, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS), name="lstm_input_layer_0"),
      tf.keras.layers.CuDNNLSTM(NUM_RNN_1_UNITS, return_sequences=True, stateful=True, name="lstm_layer_1"),
      tf.keras.layers.CuDNNLSTM(NUM_RNN_1_UNITS, return_sequences=True, stateful=True, name="lstm_layer_2"),
      tf.keras.layers.CuDNNLSTM(NUM_RNN_3_UNITS, return_sequences=True, stateful=True, name="lstm_layer_3"),
      tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS, activation='relu', name="dense_output_layer_4")
    ])
  elif version == "0.5":
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=True, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS), name="lstm_input_layer_0"),
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=True, name="lstm_layer_1"),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True, name="lstm_layer_2"),
      tf.keras.layers.CuDNNLSTM(256, return_sequences=True, stateful=True, name="lstm_layer_3"),
      tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS, activation='relu', name="dense_layer_4"),
      tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS, activation='relu', name="dense_output_layer_5")
    ])
    optimizer = tf.train.AdamOptimizer()
  elif version == "0.6":
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=True, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=True),
      tf.keras.layers.CuDNNLSTM(512,  return_sequences=True, stateful=True),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS, activation='relu')
    ])
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
  elif version == "0.7":
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS, activation='relu')
    ])
    optimizer = tf.train.AdamOptimizer(learning_rate=0.000001)
  elif version == "0.8":
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=True, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS)
    ])
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
  elif version == "0.9":
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=True),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS, activation='relu')
    ])
    optimizer = tf.train.AdamOptimizer()
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.95, beta2=0.999, epsilon=1e-7)


  model.compile(optimizer, loss=loss, metrics=['accuracy'])

  if checkpoint_base_dir != None:
    # Load the most recent checkpoint
    checkpoint_dir_path = get_latest_checkpoint_dir(checkpoint_base_dir)
    latest_checkpoint_filepath = tf.train.latest_checkpoint(str(checkpoint_dir_path))
    if latest_checkpoint_filepath:
      model.load_weights(latest_checkpoint_filepath)
      print("Loaded model weights from latest checkpoint file: " + latest_checkpoint_filepath)
    else:
      print("Failed to find latest checkpoint file.")

  return model

def loss(targets, predictions):
  return tf.keras.losses.MSE(tf.cast(targets, dtype=tf.float32), tf.cast(predictions, dtype=tf.float32))

def split_input_target(chunk):
  input_samples = chunk[:-1]
  target_samples = chunk[1:]
  return input_samples, target_samples


def build_dataset_from_spectrogram(spectrogramFilePath):
  S = np.transpose(np.load(spectrogramFilePath)) 
  s_dataset = tf.data.Dataset.from_tensor_slices(S)
  sequences = s_dataset.batch(SEQ_LENGTH+1, drop_remainder=True)
  dataset = sequences.map(split_input_target)
  batchedDataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
  examples_per_epoch = len(S)//SEQ_LENGTH

  '''
  print(dataset)  #shape: (SEQ_LENGTH, n_mels)
  for input_example, target_example in dataset.take(1):
    print ('Input data: ' + str(input_example[1]))
    print ('Target data:' + str(target_example[0]))
  '''
  #print(batchedDataset) #shape: (BATCH_SIZE, SEQ_LENGTH, n_mels)
  return examples_per_epoch, batchedDataset

def build_dataset_from_spectrograms(spec_file_paths):
  all_S = None
  s_len = 0
  for spec_file_path in spec_file_paths:
    S = np.transpose(np.load(spec_file_path))
    s_len += len(S)
    s_dataset = tf.data.Dataset.from_tensor_slices(S)
    if all_S == None:
      all_S = s_dataset
    else:
      all_S.concatenate(s_dataset)

  examples_per_epoch = s_len//SEQ_LENGTH
  sequences = all_S.batch(SEQ_LENGTH+1, drop_remainder=True)
  dataset = sequences.map(split_input_target)
  batchedDataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

  return examples_per_epoch, batchedDataset

##
# Generate teh Nerkerberrckz.
#
# model  - Tensorflow RNN model, pretrained and loaded.
# seed_S - Samples of whatever mel spectrogram to get us started off
# gen_sr - Sample rate of the generated music
# gen_duration_s - Duration in seconds of the generated noise... err, music?
#
# Returns: A Mel Spectrogram (same as the return type of the librosa melspectrogram function) of terrible music.
##
def generate_terrible_mel_spectrogram(model, seed_S, gen_sr, gen_duration_s):

  num_gen_samples = math.ceil(gen_duration_s * (gen_sr / dnb_constants.MEL_N_FFT))
  gen_S = np.zeros((num_gen_samples, dnb_constants.NUM_MEL_CHANNELS))
  input_eval = tf.expand_dims(np.transpose(seed_S)[0:1024], 0)

  # Note: The model batch size should be 1
  model.reset_states()
  for i in range(num_gen_samples):
      result = model(tf.cast(input_eval, dtype=tf.float32))

      # Remove the batch dimension
      result = tf.squeeze(result, 0).numpy()
      # TODO: Normalize the result?
      input_eval = tf.expand_dims(result, 0)

      # Append the result to the generated mel spectrograph data
      #if len(gen_S) < len(result):
      #  gen_S[i:] += result[-(len(gen_S)-i):]
      #else:
      #  end = min(i + len(result), len(gen_S))
      #  gen_S[i:end] = 

      last_result = result[-1:]
      gen_S[i] = last_result

  return gen_S.transpose()