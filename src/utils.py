import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import os
import pandas as pd
from sklearn.model_selection import train_test_split
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
    x_raw_audio = []
    y_raw = []
    for token_folder in os.scandir(inp_folder_path):
        if token_folder.name in skip_folders or not token_folder.is_dir():
            continue
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

    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test, data_folder):
    train_data_file = os.path.join(data_folder, 'train_sounds.h5')
    test_data_file = os.path.join(data_folder, 'test_sounds.h5')
    with h5py.File(train_data_file, 'w') as hf:
        hf.create_dataset("train_set_x", data=X_train)
        hf.create_dataset("train_set_y", data=y_train)
    with h5py.File(test_data_file, 'w') as hf:
        hf.create_dataset("test_set_x", data=X_test)
        hf.create_dataset("test_set_y", data=y_test)
    print("Saved the data to %s and %s" % (train_data_file, test_data_file))


def main():
    X_train, X_test, y_train, y_test = prepare_data(
        "/Users/nipasuma/Projects/kaggle/tensorflow_speech_recognition/data/train/audio")
    save_data(X_train, X_test, y_train, y_test,
              "/Users/nipasuma/Projects/kaggle/tensorflow_speech_recognition/data")


if __name__ == '__main__':
    main()
