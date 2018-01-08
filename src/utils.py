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
from datetime import datetime
import scipy.io.wavfile
#from scipy.fftpack import dct
#import matplotlib.pyplot as plt

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


def vectorize_train_data(inp_folder_path, skip_folders=("_background_noise_",)):
    """
    This requires huge memory as we load the entire dataset into memory and process it
    """
    x_raw_audio = []
    y_raw = []
    classes = []
    for token_folder in os.scandir(inp_folder_path):
        if token_folder.name in skip_folders or not token_folder.is_dir():
            continue
        classes.append(token_folder.name)
        for audio_file in os.scandir(token_folder.path):
            x_raw_audio.append(load_wav_file(os.path.abspath(audio_file.path)))
            y_raw.append(token_folder.name)
        print("%s: Completed processing %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), token_folder.name))

    print("%s: Vectorized all wav files" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Assumption: padding with zeros to handle audio clips of unequal lengths
    df = pd.DataFrame(x_raw_audio, dtype=float).fillna(AUDIO_PADDING)
    print("%s: Completed padding" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    x_raw_audio = None
    gc.collect()  # free up some memory

    X = np.array(df)
    print("%s: Computed X" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df = None
    gc.collect()

    Y = np.array(y_raw, dtype='|S9')  # to binary strings to persist on disk

    classes = np.array(classes, dtype='|S9')  # to binary strings to persist on disk

    return X, Y, classes

def get_features_from_all_files(input_folder_path, skip_folders=("_background_noise_",)):
    # Params
    preemphasis_alpha = 0.97
    frame_len_in_secs = 0.02
    frame_step_in_secs = 0.01
    NFFT = 512
    nFilts = 40
    n_leftFrames = 23
    n_rightFrames = 8
    max_nframes = 98
    nclasses = 30
    classes = []
    #class_count = 0
    #sample_count = 0
    X = []
    y = []
    for token_folder in os.scandir(input_folder_path):
        if token_folder.name in skip_folders or not token_folder.is_dir():
            continue
        classes.append(token_folder.name)
        for audio_file in os.scandir(token_folder.path):
            audio_file = os.path.abspath(audio_file.path)
            if not (audio_file).endswith('.wav'):
                continue
            X.append(extract_mel_filter_bank_features(audio_file, preemphasis_alpha, frame_len_in_secs, frame_step_in_secs, NFFT, nFilts, n_leftFrames, n_rightFrames, max_nframes))
            y.append(token_folder.name)
            #sample_count += 1
        #class_count += 1
    return X,y,classes

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


def load_data(train_data_file="../data/vectorized/sample/train_sounds.h5", test_data_file="../data/vectorized/sample/test_sounds.h5",
              classes_data_file="../data/vectorized/sample/classes_sounds.h5"):
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

def extract_mel_filter_bank_features(wavfile,preemphasis_alpha,frame_len_in_secs,frame_step_in_secs,NFFT,nFilts,n_leftFrames,n_rightFrames,max_nframes):
    """
    
    :param: wavfile,preemphasis_alpha,frame_len_in_secs,frame_step_in_secs,NFFT,nFilts,n_leftFrames,n_rightFrames,max_nframes
    :return: mel_filter_bank_features
    """

    # Read the signal
    fs, signal = scipy.io.wavfile.read(wavfile)
    signal = signal / float(2 ** 15)

    # PreEmphasis
    preemaphasized_signal = np.append(signal[0], signal[1:] - preemphasis_alpha * signal[:-1])

    # Framing
    frame_len_in_samples = int(round(frame_len_in_secs * fs))
    frame_step_in_samples = int(round(frame_step_in_secs * fs))
    signal_len = preemaphasized_signal.size
    nFrames = int(np.ceil((signal_len - frame_len_in_samples) / frame_step_in_samples))
    n_zero_padding = nFrames * frame_step_in_samples + frame_len_in_samples - signal_len
    preemaphasized_signal_padded = np.pad(signal, (0, n_zero_padding), 'constant', constant_values=0)
    indices = np.tile(np.arange(0, frame_len_in_samples), (nFrames, 1)) + np.tile(
        np.arange(0, nFrames * frame_step_in_samples, frame_step_in_samples), (frame_len_in_samples, 1)).T
    frames = preemaphasized_signal_padded[indices.astype(np.int32, copy=False)]

    # Windowing
    frames *= np.hamming(frame_len_in_samples)

    # FFT and Power Spectrum
    mag_frames = np.abs(np.fft.rfft(frames, NFFT))
    power_spect_frames = (1.0 / NFFT) * (mag_frames ** 2)

    # Filter Banks
    mel_low = 0
    mel_high = 2595.0 * np.log10(1.0 + fs / (2 * 700.0))
    mel_bins = np.linspace(mel_low, mel_high, nFilts + 2)
    f_bins = 700.0 * (10 ** (mel_bins / 2595.0) - 1)
    f_ind = np.floor((NFFT + 1) * f_bins / fs)
    f_actual_bins = np.linspace(0, fs, NFFT)
    f_actual_bins = f_actual_bins[range(int(NFFT / 2.0 + 1))]

    filter_bank_coeffs = np.zeros((nFilts, int(NFFT / 2.0 + 1)))

    for i in range(1, nFilts + 1):
        f_prev_ind = int(f_ind[i - 1])
        f_next_ind = int(f_ind[i + 1])
        f_cur_ind = int(f_ind[i])

        filter_bank_coeffs[i - 1, f_prev_ind:f_cur_ind] = (np.arange(f_prev_ind, f_cur_ind) - f_ind[i - 1]) / (
        f_ind[i] - f_ind[i - 1])
        filter_bank_coeffs[i - 1, f_cur_ind:f_next_ind] = (f_ind[i + 1] - np.arange(f_cur_ind, f_next_ind)) / (
        f_ind[i + 1] - f_ind[i])


    # Filter Bank Features
    filter_bank_features = np.dot(power_spect_frames, filter_bank_coeffs.T)
    filter_bank_features = np.where(filter_bank_features == 0, np.finfo(float).eps, filter_bank_features)
    filter_bank_features = 20 * np.log10(filter_bank_features)
    filter_bank_features_normalized = filter_bank_features - np.mean(filter_bank_features, axis=0)

    # Filter Bank Features with adjacent frames
    filter_bank_features_normalized_final = np.zeros(
        (max_nframes - n_leftFrames - n_rightFrames, nFilts * (n_leftFrames + n_rightFrames + 1)))
    for i in np.arange(n_leftFrames, nFrames - n_rightFrames):
        filter_bank_features_normalized_final[i:] = filter_bank_features_normalized[
                                                    i - n_leftFrames:i + n_rightFrames + 1, :].flatten()

    return filter_bank_features_normalized_final


def prepare_sample(h5_inp_folder, output_folder, train_sample_size=1000, test_sample_size=60):
    """
    Samples a h5py file generated from a audio vector train data set
    :param h5_inp_folder: folder containing train_sounds, test_sounds and classes_sounds vectorized h5 files
    :param output_folder: directory where to save the sampled files
    :param train_sample_size: number of examples as training data in the output
    :param test_sample_size: number of examples as test data in the output
    :return:
    """
    train_data_file = os.path.join(h5_inp_folder, 'train_sounds.h5')
    test_data_file = os.path.join(h5_inp_folder, 'test_sounds.h5')
    classes_data_file = os.path.join(h5_inp_folder, 'classes_sounds.h5')
    X, Y, _, _, classes = load_data(train_data_file, test_data_file, classes_data_file)
    print("%s: Vectorized training data" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))
    X_mini, y_mini = next(random_mini_batches(X, Y, train_sample_size + test_sample_size))  # fetch the first batch
    test_size = float(test_sample_size) / (test_sample_size + train_sample_size)
    X_train, X_test, y_train, y_test = train_test_split(X_mini, y_mini, test_size=test_size, random_state=42)
    save_data(X_train, X_test, y_train, y_test, classes, output_folder)


def main():
    # X, Y, classes = vectorize_train_data("../data/train/audio")
    # x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes = load_data()

if __name__ == '__main__':
    main()
