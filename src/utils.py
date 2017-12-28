import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import numpy as np
import h5py
import gc

AUDIO_PADDING = 0  # padding value for audio vectors to make them of equal length


def load_wav_file(filename):
    """
    Credits: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/input_data.py
    Loads an audio file and returns a float PCM-encoded array of samples.

    Args:
      filename: Path to the .wav file to load.

    Returns:
      Numpy array holding the sample data as floats between -1.0 and 1.0.
    """
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        return sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def prepare_data(inp_folder_path, skip_folders=("_background_noise_",)):
    """
    This requires huge memory as we load the entire dataset into memory and process it
    """
    x_raw_audio = []
    y_raw = []
    classes = []
    for token_folder in os.scandir(inp_folder_path):
        if token_folder.name in skip_folders or not token_folder.is_dir():
            continue
        classes.append(token_folder)
        for audio_file in os.scandir(token_folder):
            x_raw_audio.append(load_wav_file(os.path.abspath(audio_file.path)))
            y_raw.append(token_folder.name)
        print("Completed processing %s" % token_folder.name)

    # Assumption: padding with zeros to handle audio clips of unequal lengths
    df = pd.DataFrame(x_raw_audio, dtype=float).fillna(AUDIO_PADDING)
    print("Completed padding")
    x_raw_audio = None
    gc.collect()  # free up some memory

    X = np.array(df)
    print("Computed X")
    df = None
    gc.collect()

    Y = np.array(y_raw, dtype='|S9')  # to binary strings to persist on disk
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    print("Finished splitting all data to train-test")
    X = None
    y = None
    gc.collect()

    classes = np.array(classes, dtype='|S9')  # to binary strings to persist on disk

    return X_train, X_test, y_train, y_test, classes


def save_data(X_train, X_test, y_train, y_test, classes, data_folder):
    train_data_file = os.path.join(data_folder, 'train_sounds.h5')
    test_data_file = os.path.join(data_folder, 'test_sounds.h5')
    classes_data_file = os.path.join(data_folder, 'classes_sounds.h5')
    with h5py.File(train_data_file, 'w') as hf:
        hf.create_dataset("train_set_x", data=X_train)
        hf.create_dataset("train_set_y", data=y_train)
    with h5py.File(test_data_file, 'w') as hf:
        hf.create_dataset("test_set_x", data=X_test)
        hf.create_dataset("test_set_y", data=y_test)
    with h5py.File(classes_data_file, 'w') as hf:
        hf.create_dataset("classes", data=classes)
    print("Saved the data to %s, %s and %s" % (train_data_file, test_data_file, classes_data_file))


def load_data(train_data_file="../data/train_sounds.h5",
              test_data_file="../data/test_sounds.h5",
              classes_data_file="../data/classes_sounds.h5"):
    with h5py.File(train_data_file, 'r') as hf:
        x_train_orig = hf["train_set_x"][:]
        y_train_orig = hf["train_set_y"][:]
    with h5py.File(test_data_file, 'r') as hf:
        x_test_orig = hf["test_set_x"][:]
        y_test_orig = hf["test_set_y"][:]
    with h5py.File(classes_data_file, 'r') as hf:
        classes = hf["classes"][:]

    # re-shape to perform convolution operations
    x_train_orig = x_train_orig.reshape(x_train_orig.shape[0], x_train_orig.shape[1], 1, 1)
    x_test_orig = x_test_orig.reshape(x_test_orig.shape[0], x_test_orig.shape[1], 1, 1)
    y_train_orig = y_train_orig.reshape(y_train_orig.shape[0], 1)
    y_test_orig = y_test_orig.reshape(y_test_orig.shape[0], 1)

    return x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes


def convert_to_one_hot(Y, classes):
    return label_binarize(Y, classes=classes)


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- to initiate the randomness. Mostly used for testing.

    Returns:
    mini_batches -- generator of synchronous (mini_batch_X, mini_batch_Y) to be memory efficient
    """

    m = X.shape[0]  # number of training examples
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitioning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        yield (mini_batch_X, mini_batch_Y)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        yield (mini_batch_X, mini_batch_Y)


def main():
    # X_train, X_test, y_train, y_test, classes = prepare_data("../data/train/audio")
    # save_data(X_train, X_test, y_train, y_test, classes,  "../data")
    x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes = load_data()


if __name__ == '__main__':
    main()
