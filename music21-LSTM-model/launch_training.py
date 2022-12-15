import functions
from functions import get_notes, prepare_sequences,\
    create_network, train, train_network

import os
from typing import Dict, List


################################
########PARAMETERS##############
################################

from dotenv import load_dotenv
load_dotenv()

DATA_PATH = os.environ.get('DATA_PATH')
SEQ_LENGTH = int(os.environ.get('SEQ_LENGTH'))
PERIOD = int(os.environ.get('PERIOD'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
EPOCHS = int(os.environ.get('EPOCHS'))
NUM_NOTES_TO_GENERATE =int(os.environ.get('NUM_NOTES_TO_GENERATE'))
PROJECT_NAME = os.environ.get('PROJECT_NAME')
DATASET_NAME = os.environ.get('DATASET_NAME')
STORAGE_PATH = os.environ.get('STORAGE_PATH')

#load files in
notes = get_notes()

# # get amount of pitch names
n_vocab = len(set(notes))
print(f"n_vocab: {n_vocab}")

#check notes format
print(notes[:5])

#train model
train_network(notes, n_vocab)
