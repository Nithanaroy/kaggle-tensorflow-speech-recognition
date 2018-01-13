# coding: utf-8

# # Audio Recognition using Tensorflow
# 
# This approach uses CNN to build a classifier for audio inputs

# ## Import necessary modules

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from utils import *
from datetime import datetime
from tensorflow.python.client import timeline  # for profiling
from math import ceil
import os
import traceback

# A way to hint the trainer to stop training on the next epoch
STOP_TRAINING_ON_NEXT_EPOCH = "STOP_TRAINING_ON_NEXT_EPOCH"


# ## Create input placeholders
# 
# Tensorflow placeholders for X and Y. These will be dynamically set during batch G.D at runtime

def create_placeholders(n_l, n_h, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_l -- scalar, first dimension of a audio vector
    n_h -- scalar, second dimension of a audio vector
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_l, n_h, 1] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    X = tf.placeholder(tf.float32, shape=(None, n_l, n_h, 1), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, n_y), name="Y")

    return X, Y


# X, Y = create_placeholders(500, 20)
# print ("X = " + str(X))
# print ("Y = " + str(Y))


# ## Initialize Parameters
# 
# With tensorflow we only need to initialize parameters for Conv layers. Fully connected layers' paramaters are completed handled by the framework.

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
    W1 : [4, 1, 1, 8]
    W2 : [2, 1, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [4, 1, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 1, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1, "W2": W2}

    return parameters


# tf.reset_default_graph()
# with tf.Session() as sess_test:
#     parameters = initialize_parameters()
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#     print("W1 = " + str(parameters["W1"].eval()[0,0,0]))
#     print("W2 = " + str(parameters["W2"].eval()[0,0,0]))


# ## Forward Propagation
# 
# CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
# 
# Following are the parameters for all the layers:
#     - Conv2D: stride 1, padding is "SAME"
#     - ReLU
#     - Max pool: 8 by 1 filter size and an 8 by 1 stride, padding is "SAME"
#     - Conv2D: stride 1, padding is "SAME"
#     - ReLU
#     - Max pool: 4 by 1 filter size and a 4 by 1 stride, padding is "SAME"
#     - Flatten the previous output.
#     - FULLYCONNECTED (FC) layer: outputs 30 classes one for each audio utterance

def forward_propagation2(X):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing parameters "W1", "W2"
    the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    #     W1 = parameters['W1']
    #     W2 = parameters['W2']
    regularizer1 = tf.contrib.layers.l2_regularizer(scale=0.001)
    regularizer2 = tf.contrib.layers.l2_regularizer(scale=0.01)
    regularizer3 = tf.contrib.layers.l2_regularizer(scale=0.1)
    regularizer4 = tf.contrib.layers.l2_regularizer(scale=1.0)
    regularizer5 = tf.contrib.layers.l2_regularizer(scale=10.0)
    regularizer6 = None

    Z1 = tf.layers.conv2d(X, 64, (20, 8), strides=[1, 1], padding='SAME',
                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),
                          kernel_regularizer=regularizer6, name="z1")
    A1 = tf.nn.relu(Z1, name="a1")
    P1 = tf.nn.max_pool(A1, ksize=[1, 1, 3, 1], strides=[1, 2, 1, 1], padding='SAME', name="p1")
    Z2 = tf.layers.conv2d(P1, 64, (10, 4), strides=[1, 1], padding='SAME',
                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),
                          kernel_regularizer=regularizer6, name="z2")
    A2 = tf.nn.relu(Z2, name="a2")
    P2 = tf.nn.max_pool(A2, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name="p2")
    P3 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P3, 128, activation_fn=None, weights_regularizer=regularizer6)
    A3 = tf.nn.relu(Z3, name="a3")
    Z4 = tf.contrib.layers.fully_connected(A3, 30, activation_fn=None, weights_regularizer=regularizer6)
    return Z4


def forward_propagation(X):
    regularizer1 = tf.contrib.layers.l2_regularizer(scale=0.001)
    regularizer2 = tf.contrib.layers.l2_regularizer(scale=0.01)
    regularizer3 = tf.contrib.layers.l2_regularizer(scale=0.1)
    regularizer4 = tf.contrib.layers.l2_regularizer(scale=1.0)
    regularizer5 = tf.contrib.layers.l2_regularizer(scale=10.0)
    regularizer6 = None

    Z1 = tf.layers.conv2d(X, 8, (4, 1), strides=[1, 1], padding='SAME',
                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),
                          kernel_regularizer=regularizer2, name="z1")
    A1 = tf.nn.relu(Z1, name="a1")
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 1, 1], strides=[1, 8, 1, 1], padding='SAME', name="p1")
    Z2 = tf.layers.conv2d(P1, 16, (2, 1), strides=[1, 1], padding='SAME',
                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),
                          kernel_regularizer=regularizer3, name="z2")
    A2 = tf.nn.relu(Z2, name="a2")
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME', name="p2")
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 30, activation_fn=None, weights_regularizer=regularizer4)
    return Z3


# tf.reset_default_graph()
# with tf.Session() as sess:
#     np.random.seed(1)
#     X, Y = create_placeholders(64, 5)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     a = sess.run(Z3, {X: np.random.randn(2,64,1,1), Y: np.random.randn(2,5)})
#     print("Z3 = " + str(a))


# ## Compute Cost
# 
# Using the last layer Z3, compute softmax and J

def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (30, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y, name="L"), name="J")

    return cost


# tf.reset_default_graph()
# with tf.Session() as sess:
#     np.random.seed(1)
#     X, Y = create_placeholders(64, 30)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     a = sess.run(cost, {X: np.random.randn(4,64,1,1), Y: np.random.randn(4,30)})
#     print("cost = " + str(a))


# ## Define Model Accuracy

def model_accuracy(X, Y, Z3, X_pl, Y_pl, minibatch_size=64, percent_data=100, print_progress=True):
    """
    percent_data-- approximate max % amount of data to consider while computing accuracy
    """
    predict_op = tf.argmax(Z3, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y_pl, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    num_minibatches = 0
    acc_accuracy = 0  # accumulated accuracy across mini batches
    total_minibatches = ceil(X.shape[0] / float(minibatch_size))
    max_num_minibatches = total_minibatches * percent_data / 100.0
    minibatches = random_mini_batches(X, Y, minibatch_size)
    for minibatch in minibatches:
        (minibatch_X, minibatch_Y) = minibatch
        acc_accuracy += accuracy.eval({X_pl: minibatch_X, Y_pl: minibatch_Y})
        num_minibatches += 1

        if print_progress and num_minibatches % 25 == 0:
            print("%s: Accuracy after %ith batch: %f" % (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), num_minibatches, acc_accuracy / num_minibatches))
        if num_minibatches >= max_num_minibatches:
            break

    accuracy_value = acc_accuracy / num_minibatches

    return accuracy_value


## Model

# Connects all the functions and sets up training with mini batches

### Start a session


# ### Tensorboard Static Settings


def create_model(X_train, Y_train, learning_rate, forward_prop_handler=forward_propagation2):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 16000, 1, 1)
    Y_train -- test set, of shape (None, n_y = 30)
    X_test -- training set, of shape (None, 16000, 1, 1)
    Y_test -- test set, of shape (None, n_y = 30)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    m, n_l, n_h, _ = X_train.shape
    n_y = Y_train.shape[1]

    X, Y = create_placeholders(n_l, n_h, n_y)
    Z3 = forward_prop_handler(X)
    cost = compute_cost(Z3, Y)

    # Backpropagation: Using AdamOptimizer to minimize the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="adam").minimize(cost, name="adam_minimize")

    # streaming_cost, streaming_cost_update = tf.contrib.metrics.streaming_mean(cost)
    # streaming_cost_scalar = tf.summary.scalar('streaming_cost', streaming_cost)
    # tf.summary.scalar('current_cost', cost)
    # summary = tf.summary.merge_all()

    return Z3, X, Y, cost, optimizer


def run_model(X_train, Y_train, X_test, Y_test, X, Y, cost, optimizer, sess, training_writer, testing_writer,
              learning_rate=0.011, minibatch_size=64, num_epochs=100, print_cost=True):
    seed = 3  # to keep results consistent (numpy seed)
    m = X_train.shape[0]
    num_minibatches = int(np.ceil(m / float(minibatch_size)))
    training_costs = []
    testing_costs = []

    # Run the initialization
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    if print_cost:  # print the header
        print("Timestamp\t\tEpoch\tTraining Cost\tTesting Cost")

    # Do the training loop
    for epoch in range(num_epochs):

        if stop_early():
            break

        minibatch_cost = 0.
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
        for iminibatch, minibatch in enumerate(minibatches):
            (minibatch_X, minibatch_Y) = minibatch
            # IMPORTANT: The line that runs the graph on a minibatch.
            # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
            _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
            minibatch_cost += temp_cost / num_minibatches
            if print_cost and epoch == 0 and iminibatch % 10 == 0:
                print(" %s\t%i.%i\t%f\t%s" % (
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, iminibatch, minibatch_cost, "--"))

        # Print the cost every # epochs
        if print_cost and epoch % 2 == 0:
            training_costs.append(minibatch_cost)
            temp_cost = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
            testing_costs.append(temp_cost)
            print("%s\t%i\t%f\t%f" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, minibatch_cost, temp_cost))

            # Write to tensorboard
            training_summary = tf.Summary(value=[tf.Summary.Value(tag="cost", simple_value=minibatch_cost)])
            testing_summary = tf.Summary(value=[tf.Summary.Value(tag="cost", simple_value=temp_cost)])
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

    # plot_cost_test_train(len(training_costs), training_costs, testing_costs, "Learning rate = %s" % learning_rate)
    # plot_cost_test_train_same_scale(num_epochs, training_costs, testing_costs, "Learning rate = %s" % learning_rate)
    return training_costs, testing_costs


def stop_early():
    # For early manual stopping using the global environment variable
    msg_template = "I see you asked me to stop training using the environment variable, '%s'. Are you sure you want me to stop [yes/no]? "
    if STOP_TRAINING_ON_NEXT_EPOCH in os.environ and os.environ[STOP_TRAINING_ON_NEXT_EPOCH] == "Y":
        while True:
            response = input(msg_template % (STOP_TRAINING_ON_NEXT_EPOCH,))
            if response == "yes":
                print("Stopped training. Continuing with next steps")
                return True
            elif response == "no":
                print(
                    "Continuing to train. I'll ask you again if you set the value of the enviroment variable, '%s' to Y before starting the next epoch." % (
                        STOP_TRAINING_ON_NEXT_EPOCH,))
                return False
            else:
                print("Could not understand what you said. Type yes if you want me to stop training, else no.")


# ### Save profiling data to disk

# Create the Timeline object, and write it to a json file
def save_profiling_data_to_disk(run_metadata, run_name):
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('../profiling_data/%s.json' % (run_name,), 'w') as f:
        f.write(chrome_trace)


# ## Inference
# 
# - Convert audio file to vector and reshape
# - Do forward prop
# - Find the maximal class
# - Remap index to class name

def inference(audio_file, sess, X, classes, Z3):
    ra = load_wav_file(os.path.abspath(audio_file))
    x = ra.reshape(1, ra.shape[0], 1, 1)
    y_hat = tf.argmax(Z3, 1)
    prediction = sess.run(y_hat, feed_dict={X: x})
    return classes[prediction[0]]


def save_model_to_disk(run_name, saver, sess):
    os.makedirs("../saved_models/%s" % (run_name,))
    saved_path = saver.save(sess, "../saved_models/%s/%s.ckpt" % (run_name, run_name))
    print("%s: Saved model to disk at %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), saved_path))


def start_tf_session(gpu_count=0):
    tf.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    # Start an interactive session
    config = tf.ConfigProto(device_count={'GPU': gpu_count})
    sess = tf.InteractiveSession(config=config)
    # sess = tf.InteractiveSession()
    # ### Profiling
    tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    return sess


def explore_data(X_train, Y_train, X_test, Y_test, classes):
    # ## Explore the dataset

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    print("classes shape: " + str(classes.shape))


def load_data_helper(data_dir="../data/vectorized/90_10_split_from_train2/",
                     load_data_handler=load_data,
                     one_hot_encoding_handler=convert_strings_to_one_hot):
    # ## Import the dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_data_handler(
        os.path.join(data_dir, "train_sounds.h5"),
        os.path.join(data_dir, "test_sounds.h5"),
        os.path.join(data_dir, "classes_sounds.h5"))
    X_train = X_train_orig
    X_test = X_test_orig
    Y_train = one_hot_encoding_handler(Y_train_orig, classes)
    Y_test = one_hot_encoding_handler(Y_test_orig, classes)
    return X_train, Y_train, X_test, Y_test, classes


def train_from_scratch(data_dir, load_data_handler, one_hot_encoding_handler,
                       forward_prop_handler=forward_propagation,
                       learning_rate=0.001, num_epochs=25, minibatch_size=256,
                       run_name="Run at %s " % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),),
                       minibatch_size_for_accuracy=256):
    # get_ipython().magic('matplotlib notebook')
    np.random.seed(1)

    X_train, Y_train, X_test, Y_test, classes = load_data_helper(data_dir, load_data_handler, one_hot_encoding_handler)
    explore_data(X_train, Y_train, X_test, Y_test, classes)

    sess = start_tf_session(gpu_count=0)
    run_metadata = tf.RunMetadata()  # enable profiling hooks before running the model

    if not run_name:
        run_name = "N_alp-%s_batchsz-%s_ep-%s" % (learning_rate, minibatch_size, num_epochs)

    Z3, X, Y, cost, optimizer = create_model(X_train, Y_train, learning_rate=learning_rate,
                                             forward_prop_handler=forward_prop_handler)

    saver = tf.train.Saver()  # create a saver for saving variables to disk after creating the model

    # Define Tensor Board Settings after creating the graph

    training_writer = tf.summary.FileWriter("../logs/{}/training".format(run_name), sess.graph)
    testing_writer = tf.summary.FileWriter("../logs/{}/testing".format(run_name), sess.graph)

    training_costs, testing_costs = run_model(X_train, Y_train, X_test, Y_test, X, Y, cost, optimizer, sess,
                                              training_writer, testing_writer,
                                              learning_rate=learning_rate, minibatch_size=minibatch_size,
                                              num_epochs=num_epochs)

    # meta_graph_def = tf.train.export_meta_graph(filename='../saved_models/my-cnn-tf-model.meta')
    save_model_to_disk(run_name, saver, sess)
    save_profiling_data_to_disk(run_metadata, run_name)

    print(sess.list_devices())

    try:
        # test accuracy
        acc = model_accuracy(X_test, Y_test, Z3, X, Y, minibatch_size=minibatch_size_for_accuracy)
        print("%s: Test accuracy = %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), acc))
    except:
        print("%s: Exception while computing test accuracy" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))
        traceback.print_exc()

    try:
        # train accuracy
        acc = model_accuracy(X_train, Y_train, Z3, X, Y, minibatch_size=minibatch_size_for_accuracy)
        print("%s: Training accuracy = %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), acc))
    except:
        print("%s: Exception while computing train accuracy" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))
        traceback.print_exc()

    # ---------------
    # print(inference("../data/train/audio/bed/0a7c2a8d_nohash_0.wav", Z3))  # bed
    # print(inference("../data/train/audio/down/0a7c2a8d_nohash_0.wav", Z3))  # down
    # print(inference("../data/test/audio/clip_0000adecb.wav", Z3))  # happy

    return training_costs, testing_costs


def restore_model(ckpt_file="../saved_models/trained_model.ckpt", learning_rate=0.001):
    sess = start_tf_session()
    X_train, Y_train, X_test, Y_test, classes = load_data_helper()
    explore_data(X_train, Y_train, X_test, Y_test, classes)
    Z3, X, Y, cost, optimizer = create_model(X_train, Y_train, learning_rate=learning_rate)
    saver = tf.train.Saver()  # create a saver for saving variables to disk
    saver.restore(sess, ckpt_file)
    print("%s: Restored the model successfully" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))
    return X_train, Y_train, X_test, Y_test, classes, Z3, X, Y, cost, optimizer, learning_rate


def restore_model_and_run_accuracies(ckpt_file, learning_rate):
    X_train, Y_train, X_test, Y_test, classes, Z3, X, Y, cost, optimizer, learning_rate = restore_model(
        ckpt_file, learning_rate)
    # test accuracy
    acc = model_accuracy(X_test, Y_test, Z3, X, Y, minibatch_size=256)
    print("%s: Test accuracy = %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), acc))
    # train accuracy
    acc = model_accuracy(X_train, Y_train, Z3, X, Y, minibatch_size=256)
    print("%s: Training accuracy = %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), acc))


def main():
    learning_rate = 0.001
    num_epochs = 10
    minibatch_size = 256
    run_name = "N_indexedY_alp-%s_batchsz-%s_ep-%s_l1-0.01_l2-0.1_l3-1" % (learning_rate, minibatch_size, num_epochs)
    train_from_scratch(data_dir="../data/vectorized/90_10_split_from_train2/",
                       load_data_handler=load_data,
                       one_hot_encoding_handler=convert_to_one_hot,
                       forward_prop_handler=forward_propagation,
                       learning_rate=learning_rate, num_epochs=num_epochs, minibatch_size=minibatch_size,
                       run_name=run_name,
                       minibatch_size_for_accuracy=256)


if __name__ == '__main__':
    if STOP_TRAINING_ON_NEXT_EPOCH in os.environ:
        print("\nReminding you the value of the environment variable '%s' = %s\n",
              (STOP_TRAINING_ON_NEXT_EPOCH, os.environ[STOP_TRAINING_ON_NEXT_EPOCH]))
    main()
