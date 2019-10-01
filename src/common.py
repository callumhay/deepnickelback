import tensorflow as tf
import numpy as np
from pathlib import Path

import math
import random

import dnb_constants
import data_funcs

NUM_RNN_0_UNITS = 1024   # Number of RNN nodes/units in the nth layer of the network
NUM_RNN_1_UNITS = 512
NUM_RNN_2_UNITS = 256
NUM_RNN_3_UNITS = 256

SEQ_LENGTH  = 128   # Each sample in the sequence represents approx 22050/dnb_constants.MEL_N_FFT = 0.09287982 seconds of total time in a song
SEQ_LENGTH_PLUS_ONE = SEQ_LENGTH+1
BATCH_SIZE  = 32   # Represents SEQ_LENGTH*BATCH_SIZE seconds of sampled song data #1
BUFFER_SIZE = 10000

##
# Gets the string of the latest checkpoint filepath.
##
def get_latest_checkpoint_dir(base_dirpath, version=dnb_constants.DEEP_NICKELBACK_VERSION):
  return base_dirpath.joinpath(version).resolve()

##
# Builds the RNN model for tensorflow. If the checkpoint directory (python pathlib object) is given then it will
# load the most recent checkpoint from it for the given version of the model.
##
def build_model(batch_size=BATCH_SIZE, version=dnb_constants.DEEP_NICKELBACK_VERSION, checkpoint_base_dir=None, stateful=False, compile=True):

  if checkpoint_base_dir != None:
    model_checkpoint_filepath = get_latest_checkpoint_dir(checkpoint_base_dir).joinpath(dnb_constants.FULL_MODEL_FILE_NAME).resolve()
    if model_checkpoint_filepath.exists():
      try:
        model = tf.keras.models.load_model(str(model_checkpoint_filepath))
        model.compile(tf.keras.optimizers.Adam(), loss=loss, metrics=['accuracy'])
        return model
      except:
        print("Failed to find full model checkpoint.")
    else:
      print("No full model checkpoint found.")

  model = None
  optimizer = None

  if version == "0.17s":
    #seqln 128, bs 32 trains well
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(2048, return_sequences=True, stateful=stateful, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=stateful),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=stateful),
      tf.keras.layers.CuDNNLSTM(dnb_constants.NUM_MEL_CHANNELS, return_sequences=True, stateful=stateful),
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS)),
    ])
    optimizer = tf.train.AdamOptimizer(5e-7)
  elif version == "0.18s":
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(2048, return_sequences=True, stateful=stateful, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=stateful),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=stateful),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=stateful),
      tf.keras.layers.CuDNNLSTM(512, return_sequences=True, stateful=stateful),
      tf.keras.layers.CuDNNLSTM(dnb_constants.NUM_MEL_CHANNELS, return_sequences=True, stateful=stateful),
      tf.keras.layers.CuDNNLSTM(dnb_constants.NUM_MEL_CHANNELS, return_sequences=True, stateful=stateful),
      tf.keras.layers.CuDNNLSTM(dnb_constants.NUM_MEL_CHANNELS, return_sequences=True, stateful=stateful),
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS)),
    ])
    optimizer = tf.train.AdamOptimizer(0.0001)
  elif version == "0.19s":
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(2048, return_sequences=True, stateful=stateful, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(768, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS)),
    ])
    optimizer = tf.train.AdamOptimizer(1e-8)
    # NOTE: If 0.19s doesnt work out then reuse 0.17s but add more nodes on each layer and add dropout after each LSTM layer

  elif version == "0.20s":
    model = tf.keras.Sequential([
      tf.keras.layers.CuDNNLSTM(2200, return_sequences=True, stateful=stateful, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(1100, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(550, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(dnb_constants.NUM_MEL_CHANNELS, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS)),
    ])
    optimizer = tf.train.AdamOptimizer(3e-6)
  elif version == "0.21s":
    # This is the fastest learning network so far
    model = tf.keras.Sequential([
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS), batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.CuDNNLSTM(2500, return_sequences=True, stateful=stateful, batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS)),
    ])
    optimizer = tf.train.AdamOptimizer(1e-9)

  elif version == "0.22s":
    model = tf.keras.Sequential([
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS), batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS), input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.CuDNNLSTM(2048, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(dnb_constants.NUM_MEL_CHANNELS, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS)),
    ])
    optimizer = tf.train.AdamOptimizer(5e-10)
  '''
  elif version == "0.23s":
    model = tf.keras.Sequential([
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS), batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS), input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.CuDNNLSTM(2048, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(dnb_constants.NUM_MEL_CHANNELS, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS),
    ])
    optimizer = tf.train.AdamOptimizer(5e-6)
  
  elif version == "0.24s":
    model = tf.keras.Sequential([
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS), batch_input_shape=(batch_size, None, dnb_constants.NUM_MEL_CHANNELS)),
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.CuDNNLSTM(1024, return_sequences=True, stateful=stateful),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dnb_constants.NUM_MEL_CHANNELS)),
    ])
    optimizer = tf.train.AdamOptimizer()
  '''

  # NOTES:
  # Don't use a final layer of tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)),
  # Using a simple dense layer as the final layer is also not great - it trains quickly by stops short on loss

  #optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.95, beta2=0.999, epsilon=1e-7)
  
  if checkpoint_base_dir != None:
    # Load the most recent checkpoint
    checkpoint_dir_path = get_latest_checkpoint_dir(checkpoint_base_dir)
    latest_checkpoint_filepath = tf.train.latest_checkpoint(str(checkpoint_dir_path))
    if latest_checkpoint_filepath:
      model.load_weights(latest_checkpoint_filepath)
      print("Loaded model weights from latest checkpoint file: " + latest_checkpoint_filepath)
    else:
      print("Failed to find latest checkpoint file.")

  if compile:
    model.compile(optimizer, loss='mean_squared_error', sample_weight_mode="temporal", metrics=['mean_absolute_error', 'mean_squared_error'])

  return model

def loss(targets, predictions):
  return tf.keras.losses.MSE(tf.cast(targets, dtype=tf.float32), tf.cast(predictions, dtype=tf.float32))

def split_input_target(chunk):
  input_samples = chunk[:-1]
  target_samples = chunk[1:]
  return input_samples, target_samples

def build_dataset_from_spectrogram(spectrogramFilePath):
  # Load and clean up S so that there are no consecutive zero vectors at the start or end
  S = data_funcs.trim_zeros(np.transpose(np.load(spectrogramFilePath)))

  s_dataset = tf.data.Dataset.from_tensor_slices(S)
  sequences = s_dataset.batch(SEQ_LENGTH+1, drop_remainder=True)
  dataset = sequences.map(split_input_target)
  batchedDataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
  examples_per_epoch = len(S)//SEQ_LENGTH

  #print(batchedDataset) #shape: (BATCH_SIZE, SEQ_LENGTH, n_mels)
  return examples_per_epoch, batchedDataset

def generator_steps_per_epoch(spec_file_paths):
  return len(spec_file_paths)

def generator_dataset_func_from_spectrograms(spec_file_paths, batch_size):  
  file_idx = 0
  avail_file_indices = []
  while True:
    # Pick a random file
    if len(avail_file_indices) == 0:
      avail_file_indices = list(range(0, len(spec_file_paths)))
    file_idx = random.choice(avail_file_indices)
    avail_file_indices.remove(file_idx)

    #file_idx %= len(spec_file_paths)
    spec_file_path = spec_file_paths[file_idx]
    S = data_funcs.trim_zeros(np.transpose(np.load(spec_file_path)))

    # Select a random starting point for the sequences in the selected file...
    batch_data_in = []
    batch_data_out = []

    if batch_size == None:
      seq_len = min(len(S), SEQ_LENGTH)
      num_iter = len(S) // seq_len
      remainder = len(S) % seq_len
      start_idx = random.randint(0, remainder)
      for i in range(start_idx, start_idx+(seq_len*num_iter), seq_len):
        batch_data_in.append(S[i:(i+seq_len)])
        batch_data_out.append(S[(i+1):(i+seq_len+1)])
    else:
      last_possible_idx = len(S)-(SEQ_LENGTH*batch_size + 1)
      assert(last_possible_idx >= 0)
      start_idx = random.randint(0, last_possible_idx)
      for i in range(start_idx, start_idx+(SEQ_LENGTH*batch_size), SEQ_LENGTH):
        batch_data_in.append(S[i:(i+SEQ_LENGTH)])
        batch_data_out.append(S[(i+1):(i+SEQ_LENGTH_PLUS_ONE)])

    yield np.array(batch_data_in), np.array(batch_data_out)
    #file_idx += 1
    

def learning_scheduler(epoch):
  return 0.001 * tf.math.exp(0.1*epoch)

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
def generate_terrible_mel_spectrogram(model, seed_S, gen_sr, gen_duration_s, stateful=False):

  SEED_LEN_TO_COPY = 2048

  num_gen_samples = max(math.ceil(gen_duration_s * gen_sr / 512), SEED_LEN_TO_COPY)
  seed_S_t = data_funcs.trim_zeros(np.transpose(seed_S))
  gen_S = []

  model.reset_states()
  
  input_eval = np.zeros((1, SEED_LEN_TO_COPY, seed_S_t.shape[1]))
  input_eval[0, range(0,SEED_LEN_TO_COPY), :] = seed_S_t[:SEED_LEN_TO_COPY]
  #next_seq = model.predict(input_eval)

  for i in range(num_gen_samples):
    next_seq = model.predict(input_eval)

    if i == 0:
      for j in range(next_seq.shape[1]):
        gen_S.append(next_seq[0][j].copy())

    else:
      gen_S.append(next_seq[0][-1].copy())
      
    newSeq = next_seq[0][-1]
    newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))

    if stateful:
      input_eval = newSeq
    else:
      input_eval = np.concatenate((input_eval, newSeq), axis=1)
      result = np.zeros((1, input_eval.shape[1]-1, input_eval.shape[2]))
      result[0, range(0,input_eval.shape[1]-1), :] = input_eval[0, range(1,input_eval.shape[1]), :]
      input_eval = result

  return np.array(gen_S).transpose()

##
# Merge teh Nerkerberrckz with another song.
#
# model - Tensorflow RNN model, pretrained and loaded.
# og_S  - The original (not-so-terrible?) song to merge
# og_sr - The original sample rate of og_S
#
# Returns: A Mel Spectrogram (same as the return type of the librosa melspectrogram function) of terrible music merged
# from the combination of the model's generated output and the original song.
##
def merge_terrible_mel_spectrogram(model, og_S, og_sr, duration_in_s = None, seq_length=1):

  trimmed_og_S_t = data_funcs.trim_zeros(np.transpose(og_S))

  num_gen_samples = None
  if duration_in_s != None:
    num_gen_samples =  min(len(trimmed_og_S_t), math.ceil(max(1,duration_in_s) * og_sr / 512))
  else:
    num_gen_samples = len(trimmed_og_S_t)

  if seq_length < 0:
    seq_length = num_gen_samples

  num_gen_samples = max(seq_length, num_gen_samples)

  gen_S = np.zeros((num_gen_samples, dnb_constants.NUM_MEL_CHANNELS))

  ORIGINAL_SONG_WEIGHT_PERCENT = 0.01
  def weighted_avg_samples(s0, s1, w0):
    return w0*s0 + (1.0-w0)*s1

  model.reset_states()
  input_eval = np.zeros((1, seq_length, trimmed_og_S_t.shape[1]))
  input_eval[0, range(0, seq_length), :] = trimmed_og_S_t[:seq_length]

  for i in range(num_gen_samples // seq_length):
    result = model.predict(input_eval)
    np_result = tf.squeeze(result, 0).numpy()

    temp = trimmed_og_S_t[(i*seq_length + 1):(i*seq_length + seq_length + 1)]
    for j in range(len(temp)):
      temp[j] = weighted_avg_samples(temp[j], np_result[j], ORIGINAL_SONG_WEIGHT_PERCENT)

    if len(temp) < seq_length:
      temp = np.concatenate((temp, np_result[-1:]), axis=0)

    input_eval[0, range(0, seq_length), :] = temp
    
    gen_S[(i*seq_length):(i*seq_length + seq_length)] = temp

  '''
  for i in range(num_gen_samples-seq_length):

    result = model(tf.cast(input_eval, dtype=tf.float32))
    np_result = tf.squeeze(result, 0).numpy()

    temp = trimmed_og_S_t[(i+1):(i+seq_length+1)]
    for j in range(seq_length):
      temp[j] = AMPLIFY_AMOUNT*weighted_avg_samples(temp[j], np_result[j], ORIGINAL_SONG_WEIGHT_PERCENT)

    input_eval = tf.expand_dims(temp, 0)
    
    if i == 0:
      gen_S[i:(i+seq_length)] = temp
    else:
      gen_S[(i-1):(i+seq_length-1)] = (gen_S[(i-1):(i+seq_length-1)] + temp) / 2.0
  '''

  '''
    temp = trimmed_og_S_t[i+1:3+i]
    for j in range(2):
      # Find the smallest number greater than zero in the original data
      nonZeroVals = temp[j][temp[j] > 0]
      # Clean up all values that are less than the lowest non-zero value in the original data
      if len(nonZeroVals) > 0:
        minNonZeroVal = np.amin(nonZeroVals)
        cleanup_func = np.vectorize(lambda x: 0 if x < minNonZeroVal else x)
        np_result[j] = cleanup_func(np_result[j])

      temp[j] = weighted_avg_samples(temp[j], np_result[j], ORIGINAL_SONG_WEIGHT_PERCENT / float(i+1))

    input_eval = tf.expand_dims(temp, 0)
    gen_S[i] = temp[1]
  '''

  return gen_S.transpose()