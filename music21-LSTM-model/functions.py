from datetime import datetime
from typing import Dict, List
import numpy as np

import os
import pickle
import pathlib
import glob

from music21 import converter, instrument, stream, note, chord

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Bidirectional, Flatten
from keras import utils
from keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqSelfAttention

from google.cloud import storage

from dotenv import load_dotenv
load_dotenv()

DATA_SOURCE = os.environ.get('DATA_SOURCE')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
DATA_PATH = os.environ.get('DATA_PATH')
NUM_FILES = int(os.environ.get('NUM_FILES'))
DATASET=os.environ.get('DATASET')
SAVING=os.environ.get('SAVING')

def get_notes():
    """ Get all the notes and chords from the midi files - Call BEFORE train """
    notes = []

    if DATA_SOURCE == 'local' :
        print("Parsing files from local directory...")
        data_path = pathlib.Path(DATA_PATH)
        filenames = glob.glob(str(data_path/"*.mid"))

    if DATA_SOURCE == 'cloud' :
        print("Downloading and parsing files from bucket on Google Cloud ")
        dl_dir = "downloaded_files/"

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        prefix = DATASET+'/'
        blobs = bucket.list_blobs(prefix=prefix)
        iter = 0

        #Create directory if missing
        pathlib.Path(dl_dir+DATASET).mkdir(parents=True, exist_ok=True)

        for blob in blobs :
            while iter < NUM_FILES :
                filename = blob.name
                blob.download_to_filename(dl_dir+filename)
                iter +=1

        data_path = pathlib.Path(dl_dir+DATASET)
        filenames = glob.glob(str(data_path/"*.mid"))


    for file in filenames[:NUM_FILES]:
        stream_file = converter.parse(file)

        #print("Parsing %s" % file)

        components_to_parse = []
        for element in stream_file.recurse():
            components_to_parse.append(element)

        #components_to_parse = stream_file.flat.notes #return to this to preview model output 10 epochs 1st training

        for element in components_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch) + " " +  str(float(element.quarterLength)))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder) + " " + str(float(element.quarterLength)))
            elif isinstance(element, note.Rest):
                notes.append(str(element.name)  + " " + str(float(element.quarterLength)))

    if SAVING =='local' :
        with open(f"../raw_data/notes/notes_{NUM_FILES}files.pickle", 'wb') as f:
            pickle.dump(notes, f)

    if SAVING =='cloud' :
        pathlib.Path('notes_pickles').mkdir(parents=True, exist_ok=True)
        blob_name=f"notes_pickles/notes_{NUM_FILES}files.pickle"
        with open(blob_name, 'wb') as f:
            pickle.dump(notes, f)

    print(f"Parsing done on {NUM_FILES} file(s).")

    return notes

def prepare_sequences(notes, n_vocab):

    print("Preparing sequences...")

    SEQ_LENGTH = int(os.environ.get('SEQ_LENGTH'))
    PROJECT_NAME = os.environ.get('PROJECT_NAME')
    DATASET_NAME = os.environ.get('DATASET_NAME')
    STORAGE_PATH = os.environ.get('STORAGE_PATH')

    """ Prepare the sequences used by the Neural Network """
    sequence_length = SEQ_LENGTH


    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)
    network_output = utils.to_categorical(network_output)

    print("Sequences ready.")

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(Bidirectional(LSTM(512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(0.3))

    model.add(LSTM(512,return_sequences=True))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    SEQ_LENGTH = int(os.environ.get('SEQ_LENGTH'))
    NUM_FILES = int(os.environ.get('NUM_FILES'))
    PERIOD = int(os.environ.get('PERIOD'))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
    EPOCHS = int(os.environ.get('EPOCHS'))

    now = datetime.now()
    model_timestamp= now.strftime('%d-%m-%H-%M-%S')
    filepath = os.path.abspath(f"model_checkpoints_weights/{model_timestamp}/weights-1LSTMAtt1LSTMLayer-num_files_{NUM_FILES}-seq_length_{SEQ_LENGTH}-batch_size_{BATCH_SIZE}"+"-{epoch:03d}-{loss:.4f}.hdf5")
    checkpoint = ModelCheckpoint(
        filepath,
        period=PERIOD, #Save weights every xx epochs
        monitor='loss',
        verbose=1,
        save_best_only=False,
        mode='min'
    )

    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)

    if SAVING =='cloud' :
        weights_path = pathlib.Path(f'model_checkpoints_weights/{model_timestamp}')
        filenames = glob.glob(str(weights_path/"*.hdf5"))
        for filename in filenames :
            save_on_bucket(filename,filename)

        #Saving pickle file in the same folder
        pickle_filename = f"notes_pickles/notes_{NUM_FILES}files.pickle"
        pickle_path = pathlib.Path(pickle_filename.split('/')[1])
        blob_name=str((weights_path/pickle_path).stem)
        save_on_bucket(blob_name, pickle_filename)


def train_network(notes, n_vocab):
    """ Train a Neural Network to generate music """

    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def save_on_bucket(blob_name,filename) :

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(filename)
