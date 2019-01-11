"""Convolutional neural network classififer."""

# Import dependences for python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Get addition paramesters of model
def get_params():
    return {
        "drop_rate": 0.5
    }

# Define the CNN classifier model
def model(features, labels, mode, params):
    # get sequence of input features
    sequence = features["residue"]
    # get labels of input features
    labels = labels["label"]

    # set drop-out rate equal 0 if it's not traning phase
    drop_rate = params.drop_rate if mode == tf.estimator.ModeKeys.TRAIN else 0.0
    # set training variable to check if it's training phase
    training = mode == tf.estimator.ModeKeys.TRAIN

    # features initialization
    features = sequence
    # loop through all the convolution layer and add related stuff to the model
    for i, filters in enumerate([128]):
        # convolution layer with ReLU activation function 
        # and size of filters*kernel_size (128*3 in this case)
        features = tf.layers.conv2d(
            features, filters=filters, activation=tf.nn.relu, kernel_size=3, padding="same",
            name="conv_%d" % (i + 1))
        # batch-norm layer
        features = tf.layers.batch_normalization(features, training=training)
        # max-pooling layer
        features = tf.layers.max_pooling2d(
            inputs=features, pool_size=2, strides=2, padding="same",
            name="pool_%d" % (i + 1))

    # flatten layer
    features = tf.contrib.layers.flatten(features)

    # drop-out layer
    features = tf.layers.dropout(features, drop_rate)
    # dense layer with ReLU activation function
    features = tf.layers.dense(features, 32, activation=tf.nn.relu, name="dense_1")
    # batch-norm layer
    features = tf.layers.batch_normalization(features, training=training)

    # drop-out layer
    features = tf.layers.dropout(features, drop_rate)
    # dense layer (final layer) with 2 neurons, has shape [batch_size, 2]
    logits = tf.layers.dense(features, params.num_classes,
                             name="dense_2")

    # get the predictions
    predictions = tf.argmax(logits, axis=1)

    # get the cross-entropy loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)

    # return the result
    return {"predictions": predictions}, loss


# Eval metrics
def eval_metrics(unused_params):
    return {
        # get the accuracy
        "accuracy": tf.contrib.learn.MetricSpec(tf.metrics.accuracy)
    }
