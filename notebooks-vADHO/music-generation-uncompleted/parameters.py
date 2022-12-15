import pretty_midi
import numpy as np
import tensorflow as tf
import os
import pathlib
import glob

#Path to MIDI files directory
data_folder = 'chopin_28prelude_2channels' #to input as a string
data_path = os.path.join('..','..','raw_data', data_folder)
data_path = pathlib.Path(data_path)

# # Alternative though Google Colab
# data_path = pathlib.Path('drive/MyDrive/Colab Notebooks/music-generation/data/chopin_2channels')
filenames = glob.glob(str(data_path/"*.mid"))
# print('Number of files:', len(filenames))

# General parameters
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
_SAMPLING_RATE = 16000 # Sampling rate for audio playback
vocab_size = 128 # Represents the number of note pitches
key_order = ['note', 'octave', 'step', 'duration'] # Features we keep

# Hyper-parameters
seq_length = 50
batch_size = 16
note_weight= 1.0
octave_weight= 0.25
step_weight = 1.0
duration_weight = 1.0
patience = 5
epochs = 100
learning_rate = 0.0005

# Parameters used to generate the music notes
temperature = 2
file_number = 1 #Choose one file among the existing file to start the prediction
sample_file = filenames[file_number]
#example_file = 'example.midi'
#example_pm = notes_to_midi(
#    raw_notes, out_file=example_file, instrument_name=instrument_name)
pm = pretty_midi.PrettyMIDI(sample_file)
instrument = pm.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
