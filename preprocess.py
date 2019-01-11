"""FASTA preprocessing."""

# import numpy for caculating
import numpy as np
# import Bio for fasta file handling
from Bio import SeqIO
# import tensorflow 
import tensorflow as tf
# import KFold for cross-validation handling
from sklearn.model_selection import KFold

# 5-fold cross-validation
KF = KFold(n_splits=5)
# window size for sliding window technique
WINDOW_SIZE = 25
# 20 standard amino acids presentation and '@' digit present the missing residue
STANDARD_AMINO_ACID = 'ARNDCQEGHILKMFPSTWYV@'
# convert standard amino acids char list to int list
char_to_int = dict((c, i) for i, c in enumerate(STANDARD_AMINO_ACID))
# convert standard amino acids int list to char list
int_to_char = dict((i, c) for i, c in enumerate(STANDARD_AMINO_ACID))


# Residue Feature Extractor
class ResidueFeatureExtractor:
    def __init__(self, name, sequence):

        # append missing residue element to sequence
        for _ in range(int(WINDOW_SIZE / 2)):
            sequence = "@" + sequence + "@"

        # integer encode the sequence
        integer_encoded_seq = self.get_integer_values_of_sequence(sequence)

        # one hot the sequence
        onehot_encoded_seq = self.get_one_hot_sequence(integer_encoded_seq)

        # feature sequence
        residue_feature_sequence = self.get_residue_feture_sequence(onehot_encoded_seq)

        # add the attributes to self
        self.name = name
        self.sequence = sequence
        self.integer = integer_encoded_seq
        self.onehot = onehot_encoded_seq
        self.features = residue_feature_sequence

    # get integer values from a sequence
    @staticmethod
    def get_integer_values_of_sequence(sequence):
        integer_encoded = [char_to_int[char] for char in sequence]
        return integer_encoded

    # one-hot encoding an integer sequence
    @staticmethod
    def get_one_hot_sequence(integer_sequence):
        # init an empty list
        one_hot_encoded_sequence = list()
        # loop through integer sequence and flag the index of each amino acid as 1 for each amino acid
        for value in integer_sequence:
            one_hot_encoded_char = [0 for _ in range(len(STANDARD_AMINO_ACID))]
            one_hot_encoded_char[value] = 1
            one_hot_encoded_sequence.append(one_hot_encoded_char)
        return one_hot_encoded_sequence

    # convert one-hot encoded sequence to final feature sequence base on sliding-window technique
    @staticmethod
    def get_residue_feture_sequence(one_hot_encoded_sequence):
        # init an empty list
        residue_feature_sequence = list()
        # loop through the sequence
        for i in range(len(one_hot_encoded_sequence) - WINDOW_SIZE + 1):
            # each element is converted to [WINDOW_SIZE] elements around it
            residue_feature = one_hot_encoded_sequence[i: i + WINDOW_SIZE]
            # append to the final sequence
            residue_feature_sequence.append(residue_feature)
        return residue_feature_sequence


# get input features from input file path, follow the mode (training or evaluate) and cross-validation index
def get_input_features(input_file, mode, cross_val_index):
    # init an empty list
    input_features = list()
    # read the fasta sequences from input file
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')

    # loop through fasta sequences
    for fasta in fasta_sequences:
        # get name and value of each sequence
        name, sequence = fasta.id, str(fasta.seq)
        # get the ResidueFeatureExtractor object of current sequence
        extractor = ResidueFeatureExtractor(name, sequence)
        # append the feature to the list
        input_features += extractor.features

    # convert to array with data type uint8 (0-255)
    input_features = np.array(input_features, dtype=np.uint8)
    # split train-test set following k-fold cross validation
    splited_features = list(KF.split(input_features))
    # get the training set index
    train_index = splited_features[cross_val_index][0]
    # get the evaluating set index
    eval_index = splited_features[cross_val_index][1]
    
    # return training set in train mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        return input_features[train_index]
    # return evaluating set in evalutae mode
    else:
        return input_features[eval_index]


# get input labels from input file path, follow the mode (training or evaluate) and cross-validation index
def get_input_labels(input_file, mode, cross_val_index):
    # init an empty list
    input_labels = list()
    # read the fasta sequences from input file
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')
    # loop through fasta sequences
    for fasta in fasta_sequences:
        # get the value of each sequence
        sequence = str(fasta.seq)
        # append to the list
        input_labels += list(sequence)

    # convert to array with data type uint8 (0-255)
    input_labels = np.array(input_labels, dtype=np.uint8)
    # split train-test set following k-fold cross validation
    splited_labels = list(KF.split(input_labels))
    # get the training set index
    train_index = splited_labels[cross_val_index][0]
    # get the evaluating set index
    eval_index = splited_labels[cross_val_index][1]
    # return training set in train mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        return input_labels[train_index]
    # return evaluating set in train mode
    else:
        return input_labels[eval_index]


# get the test features from input file path
def get_test_features(input_file):
    # init the empty list
    input_feature = list()
    # read the fasta sequences from input file
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')

    # loop through fasta sequences
    for fasta in fasta_sequences:
        # get name and value of each sequence
        name, sequence = fasta.id, str(fasta.seq)
        # get the ResidueFeatureExtractor object of current sequence
        extractor = ResidueFeatureExtractor(name, sequence)
        # append the feature to the list
        input_feature += extractor.features
    # convert to array
    return np.array(input_feature, dtype=np.uint8)

# get the test labels from input file path
def get_test_labels(input_file):
    # init the empty list
    input_labels = list()
    # read the fasta sequences from input file
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')
    # loop through fasta sequences
    for fasta in fasta_sequences:
        # get value of each sequence
        sequence = str(fasta.seq)
        # append to the list
        input_labels += list(sequence)
    # convert to array
    return np.array(input_labels, dtype=np.uint8)