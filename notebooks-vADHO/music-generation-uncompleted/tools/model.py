import numpy as np
import tensorflow as tf

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras_self_attention import SeqSelfAttention

from tools.all_functions import midi_to_notes

from parameters import key_order,\
    seq_length, batch_size,\
    note_weight, octave_weight, step_weight, duration_weight,\
    patience, epochs, learning_rate


def create_sequences(
    dataset: tf.data.Dataset,
    seq_length: int,
    vocab_size = 128,
) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""
  seq_length = seq_length+1

  # Take 1 extra for the labels = targets = predicted notes
  # shift : spacing between two windows
  # stride : inside a given window, spacing between two notes ==> we don't want to skip notes so we take stride = 1*
  # drop_remainder : representing whether the last windows should be dropped if their size is smaller than size

  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)

  # Normalize note pitch
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0]
    return x

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

    return scale_pitch(inputs), labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

def create_train_ds_per_file(filename) :

    notes = midi_to_notes(filename)

    #converting the value of the note to its corresponding note and octave

    notes_dict = {0:0, 12:0, 24:0, 36:0, 48:0, 60:0, 72:0, 84:0, 96:0, 108:0, 120:0,
        1:1, 13:1, 25:1, 37:1, 49:1, 61:1, 73:1, 85:1, 97:1, 109:1, 121:1,
        2:2, 14:2, 26:2, 38:2, 50:2, 62:2, 74:2, 86:2, 98:2, 110:2, 122:2,
        3:3, 15:3, 27:3, 39:3, 51:3, 63:3, 75:3, 87:3, 99:3, 111:3, 123:3,
        4:4, 16:4, 28:4, 40:4, 52:4, 64:4, 76:4, 88:4, 100:4, 112:4, 124:4,
        5:5, 17:5, 29:5, 41:5, 53:5, 65:5, 77:5, 89:5, 101:5, 113:5, 125:5,
        6:6, 18:6, 30:6, 42:6, 54:6, 66:6, 78:6, 90:6, 102:6, 114:6, 126:6,
        7:7, 19:7, 31:7, 43:7, 55:7, 67:7, 79:7, 91:7, 103:7, 115:7, 127:7,
        8:8, 20:8, 32:8, 44:8, 56:8, 68:8, 80:8, 92:8, 104:8, 116:8,
        9:9, 21:9, 33:9, 45:9, 57:9, 69:9, 81:9, 93:9, 105:9, 117:9,
        10:10, 22:10, 34:10, 46:10, 58:10, 70:10, 82:10, 94:10, 106:10, 118:10,
        11:11, 23:11, 35:11, 47:11, 59:11, 71:11, 83:11, 95:11, 107:11, 119:11}

    octaves_dict = {}
    for n in range(10) :
      octave_dict = { i : n for i in range(n*12, (n+1)*12)}
      octaves_dict = {**octaves_dict, **octave_dict}

    octave_dix = { i : 10 for i in range(10*12, 128)}
    octaves_dict = {**octaves_dict, **octave_dix}

    #creating 2 new columns 'notes' (1:C, 2:C# ...) and 'octave' according to the table in slack
    notes['note'] = [notes_dict[note] for note in notes['pitch']]
    notes['octave'] = [octaves_dict[note] for note in notes['pitch']]

    #Fetching features and transform to tensorflow dataset
    train_notes = np.stack([notes[key] for key in key_order], axis=1) #Keeps columns of key_order and transform to numpy.ndarray format
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

    seq_length_temp = seq_length+1
    #Creating windows
    windows = notes_ds.window(seq_length_temp, shift=1, stride=1,
                                  drop_remainder=True)
    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length_temp, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    '''# Normalize note pitch
    def scale_pitch(x):
        x = x/[vocab_size,1.0,1.0]
        return x'''

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

        return inputs, labels

    seq_ds = sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

    return seq_ds

def generate_train_ds(filenames) :
    #Concatenate all batches of sequences by instanciating on the first file
    seq_ds_all = create_train_ds_per_file(filenames[0])

    for filename in filenames[1:] :
        seq_ds = create_train_ds_per_file(filename)
        seq_ds_all = seq_ds_all.concatenate(seq_ds)

    return seq_ds_all

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

def build_model() :

    ###################
    ###**STRUCTURE**###
    ###################
    input_shape = (seq_length, 4)
    inputs = tf.keras.Input(input_shape)
    x1 = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x2 = SeqSelfAttention(attention_activation='sigmoid')(x1)
    x3 = Bidirectional(LSTM(128, return_sequences=True))(x2)
    x4 = SeqSelfAttention(attention_activation='sigmoid')(x3)
    x5 = tf.keras.layers.Dropout(0.3)(x4)
    x6 = LSTM(128)(x5)

    outputs = {
    'note': tf.keras.layers.Dense(128, name='note')(x6),
    'octave': tf.keras.layers.Dense(128, name='octave')(x6),
    'step': tf.keras.layers.Dense(1, name='step')(x6),
    'duration': tf.keras.layers.Dense(1, name='duration')(x6),
    }


    model = tf.keras.Model(inputs, outputs)

    loss = {
          'note': tf.keras.losses.SparseCategoricalCrossentropy(
              from_logits=True),
          'octave': tf.keras.losses.SparseCategoricalCrossentropy(
              from_logits=True),
          'step': mse_with_positive_pressure,
          'duration': mse_with_positive_pressure,
    }

    ###################
    ###**COMPILING**###
    ###################
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
    loss=loss,
    loss_weights={
        'note': note_weight,
        'octave': octave_weight,
        'step': step_weight,
        'duration': duration_weight,
    },
    optimizer=optimizer,
    )

    return model

def launch_training(model, train_ds):

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=patience,
            verbose=1,
            restore_best_weights=True),
    ]

    # Ask equivalent of %%time in Python
    history = model.fit(
        train_ds,
        epochs=35,
        callbacks=callbacks,
    )

    return history

def plot_losses(history) :
    plt.plot(history.epoch, history.history['loss'], label='total');
    plt.plot(history.epoch, history.history['duration_loss'], label='duration');
    plt.plot(history.epoch, history.history['note_loss'], label='note');
    plt.plot(history.epoch, history.history['octave_loss'], label='octave');
    plt.plot(history.epoch, history.history['step_loss'], label='step');
    plt.legend()
    plt.show()

def pitch_to_note_octave(raw_notes):

    #Dictionnary for notes
    notes_dict = {0:0, 12:0, 24:0, 36:0, 48:0, 60:0, 72:0, 84:0, 96:0, 108:0, 120:0,
        1:1, 13:1, 25:1, 37:1, 49:1, 61:1, 73:1, 85:1, 97:1, 109:1, 121:1,
        2:2, 14:2, 26:2, 38:2, 50:2, 62:2, 74:2, 86:2, 98:2, 110:2, 122:2,
        3:3, 15:3, 27:3, 39:3, 51:3, 63:3, 75:3, 87:3, 99:3, 111:3, 123:3,
        4:4, 16:4, 28:4, 40:4, 52:4, 64:4, 76:4, 88:4, 100:4, 112:4, 124:4,
        5:5, 17:5, 29:5, 41:5, 53:5, 65:5, 77:5, 89:5, 101:5, 113:5, 125:5,
        6:6, 18:6, 30:6, 42:6, 54:6, 66:6, 78:6, 90:6, 102:6, 114:6, 126:6,
        7:7, 19:7, 31:7, 43:7, 55:7, 67:7, 79:7, 91:7, 103:7, 115:7, 127:7,
        8:8, 20:8, 32:8, 44:8, 56:8, 68:8, 80:8, 92:8, 104:8, 116:8,
        9:9, 21:9, 33:9, 45:9, 57:9, 69:9, 81:9, 93:9, 105:9, 117:9,
        10:10, 22:10, 34:10, 46:10, 58:10, 70:10, 82:10, 94:10, 106:10, 118:10,
        11:11, 23:11, 35:11, 47:11, 59:11, 71:11, 83:11, 95:11, 107:11, 119:11}

    #Dictionnary for octaves
    octaves_dict = {}
    for n in range(10) :
        octave_dict = { i : n for i in range(n*12, (n+1)*12)}
        octaves_dict = {**octaves_dict, **octave_dict}

    octave_dix = { i : 10 for i in range(10*12, 128)}
    octaves_dict = {**octaves_dict, **octave_dix}

    #creating the sample dataset
    raw_notes['note'] = [notes_dict[note] for note in raw_notes['pitch']]
    raw_notes['octave'] = [octaves_dict[note] for note in raw_notes['pitch']]

    return raw_notes

def predict_next_note(
    notes: np.ndarray,
    keras_model: tf.keras.Model,
    temperature: float = 1.0) -> int:
  """Generates a note IDs using a trained sequence model."""

  #assert temperature > 0

  # Add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = model.predict(inputs)
  note = predictions['note']
  octave = predictions['octave']
  #pitch_logits = predictions['pitch']
  step = predictions['step']
  duration = predictions['duration']

  #pitch_logits /= temperature
  note = tf.random.categorical(note, num_samples=1)
  octave = tf.random.categorical(octave, num_samples=1)
  note = tf.squeeze(note, axis=-1)
  octave = tf.squeeze(octave, axis=-1)
  duration = tf.squeeze(duration, axis=-1)
  step = tf.squeeze(step, axis=-1)

  # `step` and `duration` values should be non-negative
  step = tf.maximum(0, step)
  duration = tf.maximum(0, duration)

  return int(note), int(octave), float(step), float(duration)

def generate_notes(input_notes, temp, num_pred, seq_length):

    temperature = temp
    num_predictions = num_pred

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    # The initial sequence of notes; pitch is normalized similar to training
    # sequences
    input_notes = sample_notes[:seq_length]

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        note, octave, step, duration = predict_next_note(input_notes, model, temperature)
        #pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        #reconverting note and octave into pitch
        #pitch = note + octave*12
        input_note = (note, octave, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*key_order, 'start', 'end'))

    generated_notes['pitch'] = generated_notes['note'] + generated_notes['octave']*12
    generated_notes = generated_notes[['pitch', 'step', 'duration', 'start','end']]

    return generated_notes
