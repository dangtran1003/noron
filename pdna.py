"""PDNA dataset preprocessing and specifications."""

# Import dependences for python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import numpy for caculating
import numpy as np
# import tensorflow 
import tensorflow as tf
# import preprocess module
import preprocess

# datasets directory
LOCAL_DIR = "data/TargetDNA/"
# input train sequences file name
INPUT_TRAIN_SEQUENCE_FILE = 'PDNA-859_sequence.fasta'
# input train labels file name
INPUT_TRAIN_LABEL_FILE = 'PDNA-859_label.fasta'
# input test sequences file name
INPUT_TEST_SEQUENCE_FILE = 'PDNA-TEST_sequence.fasta'
# input test labels file name
INPUT_TEST_LABEL_FILE = 'PDNA-TEST_label.fasta'

# number of classes
NUM_CLASSES = 2

# Get addition dataset params.
def get_params():
    return {
        "num_classes": NUM_CLASSES,
    }

# This function will be called once to prepare the dataset.
def prepare():
    """Do some addition things here"""

#Create an instance of the dataset object.
def read(split, is_test, cross_val_index):

    # get the sequence list and labels in testing phase
    if(is_test):
        # get sequence list
        sequence = preprocess.get_test_features(LOCAL_DIR + INPUT_TEST_SEQUENCE_FILE)
        print("Loaded %d TEST residue fetures." % len(sequence))
        # get labels
        labels = preprocess.get_test_labels(LOCAL_DIR + INPUT_TEST_LABEL_FILE)
        print("Loaded %d TEST labels" % len(labels))
    # get the sequence list and labels in training or evaluating phase
    else:
        # get sequence list
        sequence = preprocess.get_input_features(LOCAL_DIR + INPUT_TRAIN_SEQUENCE_FILE, split, cross_val_index)
        print("Loaded %d residue fetures." % len(sequence))

        # get labels
        labels = preprocess.get_input_labels(LOCAL_DIR + INPUT_TRAIN_LABEL_FILE, split, cross_val_index)
        print("Loaded %d labels" % len(labels))

    # add one shape to sequence list for standardizing 
    new_shape = list(sequence.shape)
    new_shape.append(1)
    sequence = np.reshape(sequence, new_shape)

    # return the dataset with sequence list and labels
    return tf.data.Dataset.from_tensor_slices((sequence, labels))

# Parse input record to features and labels.
def parse(residue, label):
    # convert input to float
    residue = tf.to_float(residue)
    # convert lable to int
    label = tf.to_int64(label)
    # return feature and label
    return {"residue": residue}, {"label": label}
