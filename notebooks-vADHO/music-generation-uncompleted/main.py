from tools.all_functions import midi_to_notes
    # midi_to_notes,\
    # plot_piano_roll, plot_distributions, notes_to_midi,\
    # create_sequences, mse_with_positive_pressure,predict_next_note,\
    # create_train_ds_per_file,

import parameters
from tools.model import build_model,\
    launch_training, pitch_to_note_octave, generate_notes,\
    generate_train_ds

# Creating training dataset
train_ds = generate_train_ds(parameters.filenames)

# Building LSTM model
model = build_model()

# Training the model
history = launch_training(model,train_ds)

#Getting "raw notes" to input to the model to begin the prediction
raw_notes = midi_to_notes(parameters.sample_file)
raw_notes = pitch_to_note_octave(raw_notes)

#Getting predictions
predictions = generate_notes(
    raw_notes,
    parameters.temperature,
    120,
    parameters.seq_length)
