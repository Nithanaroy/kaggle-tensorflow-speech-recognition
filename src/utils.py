import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import os


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


def prepare_data(foldername, skip_folders=["_background_noise_"]):
    files = 0
    for folder in os.scandir(foldername):
        if folder in skip_folders or not folder.is_dir():
            continue
        files += len([1 for x in list(os.scandir(folder)) if x.is_file()])
    return files


def main():
    print(prepare_data("/Users/nipasuma/Projects/kaggle/tensorflow_speech_recognition/data/train/audio"))


if __name__ == '__main__':
    main()
