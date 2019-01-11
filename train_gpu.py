"""This module handles training and evaluation of a neural network model.

Invoke the following command to train the model:
python -m trainer --model=cnn --dataset=pdna

Monitor the logs on Tensorboard:
tensorboard --logdir=output"""

# Import dependences for python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import cnn module
import cnn
# import pdna mudule
import pdna

# import tensorflow 
import tensorflow as tf

# logging for visualizing training process
tf.logging.set_verbosity(tf.logging.INFO)

# Define the global arguments:
# Model name
tf.flags.DEFINE_string("model", "cnn", "Model name.")
# Dataset name
tf.flags.DEFINE_string("dataset", "pdna", "Dataset name.")
# Optional output dir
tf.flags.DEFINE_string("output_dir", "", "Optional output dir.")
# Schedule
tf.flags.DEFINE_string("schedule", "train_and_evaluate", "Schedule.")
# Hyper parameters
tf.flags.DEFINE_string("hparams", "", "Hyper parameters.")
# Number of training epochs
tf.flags.DEFINE_integer("num_epochs", 10000, "Number of training epochs.")
# Summary steps
tf.flags.DEFINE_integer("save_summary_steps", 10, "Summary steps.")
# Checkpoint steps for saving
tf.flags.DEFINE_integer("save_checkpoints_steps", 10, "Checkpoint steps.")
# Number of eval steps
tf.flags.DEFINE_integer("eval_steps", None, "Number of eval steps.")
# Eval frequency
tf.flags.DEFINE_integer("eval_frequency", 1000, "Eval frequency.")
# Cross validation index
tf.flags.DEFINE_integer("cross_val_index", 0, "Cross validation index.")
# Check for testing phase
tf.flags.DEFINE_boolean("is_test", False, "Check for testing phase.")

# get the global arguments
FLAGS = tf.flags.FLAGS

MODELS = {
    # This is a dictionary of models, the keys are model names, and the values
    # are the module containing get_params, model, and eval_metrics.
    "cnn": cnn
}

DATASETS = {
    # This is a dictionary of datasets, the keys are dataset names, and the
    # values are the module containing get_params, prepare, read, and parse.
    "pdna": pdna
}

HPARAMS = {
    # optimization option
    "optimizer": "Adam",
    # learning rate
    "learning_rate": 0.00001,
    # number of steps to decay learning rate
    "decay_steps": 5000,
    # batch size
    "batch_size": 128
}

# Aggregates and returns hyper parameters
def get_params():
    # assign hyper parameters
    hparams = HPARAMS
    # add dataset parameters
    hparams.update(DATASETS[FLAGS.dataset].get_params())
    # add model parameters
    hparams.update(MODELS[FLAGS.model].get_params())

    # convert to tf.contrib.training.HParams
    hparams = tf.contrib.training.HParams(**hparams)
    # parse to FLAGS
    hparams.parse(FLAGS.hparams)

    return hparams

# Returns an input function to read the dataset
def make_input_fn(mode, params):
    def _input_fn():
        # get dataset
        dataset = DATASETS[FLAGS.dataset].read(mode, FLAGS.is_test, FLAGS.cross_val_index)
        # refactor the dataset following number of epochs and batch size
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.repeat(FLAGS.num_epochs)
            dataset = dataset.shuffle(params.batch_size * 5)
        # convert to dataset object
        dataset = dataset.map(
            DATASETS[FLAGS.dataset].parse, num_parallel_calls=8)
        # add the batch size
        dataset = dataset.batch(params.batch_size)
        # creates an iterator for enumerating the elements of dataset.
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    return _input_fn

# Returns a model function
def make_model_fn():
    def _model_fn(features, labels, mode, params):
        # assign the model function
        model_fn = MODELS[FLAGS.model].model
        # recover step or start new step for traning
        global_step = tf.train.get_or_create_global_step()
        # get predictions and loss
        predictions, loss = model_fn(features, labels, mode, params)

        # init training optimization setting 
        train_op = None
        # in the training phase: 
        if mode == tf.estimator.ModeKeys.TRAIN:
            # get the decay function following the learning rate and decay steps
            def _decay(learning_rate, global_step):
                learning_rate = tf.train.exponential_decay(
                    learning_rate, global_step, params.decay_steps, 0.5,
                    staircase=True)
                return learning_rate

            # assign the training optimization setting
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=global_step,
                learning_rate=params.learning_rate,
                optimizer=params.optimizer,
                learning_rate_decay_fn=_decay)

        # return a ModelFnOps instance
        return tf.contrib.learn.ModelFnOps(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)

    return _model_fn

# Constructs an experiment object
def experiment_fn(run_config, hparams):
    # create the estimator
    estimator = tf.contrib.learn.Estimator(
        model_fn=make_model_fn(), config=run_config, params=hparams)
    # return an Experiment instance
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=make_input_fn(tf.estimator.ModeKeys.TRAIN, hparams),
        eval_input_fn=make_input_fn(tf.estimator.ModeKeys.EVAL, hparams),
        eval_metrics=MODELS[FLAGS.model].eval_metrics(hparams),
        eval_steps=FLAGS.eval_steps,
        min_eval_frequency=FLAGS.eval_frequency)

# Main entry point
def main(unused_argv):
    # set the model directory following the ouput_dir
    if FLAGS.output_dir:
        model_dir = FLAGS.output_dir
    # if the ouput_dir is not set, set the model directory following model name, dataset and cross validation index
    else:
        model_dir = "output/%s_%s_%s" % (FLAGS.model, FLAGS.dataset, FLAGS.cross_val_index)

    # get dataset prepared
    DATASETS[FLAGS.dataset].prepare()

    # init the session configuration
    session_config = tf.ConfigProto()
    # allow TensorFlow to automatically choose an existing and supported device 
    # to run the operations in case the specified one doesn't exist
    session_config.allow_soft_placement = True
    # allow TensorFlow use all the avilable resources of GPU 
    session_config.gpu_options.allow_growth = True
    # create a RunConfig instance
    run_config = tf.contrib.learn.RunConfig(
        model_dir=model_dir,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        save_checkpoints_secs=None,
        session_config=session_config)

    # run the learning process
    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=FLAGS.schedule,
        hparams=get_params())

# if this file run from command line (is the main program), run the app
if __name__ == "__main__":
    tf.app.run()
