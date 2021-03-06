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
from tutorial_code.input_data import AudioProcessor

# from scipy.fftpack import dct
# import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('PS')  # generate postscript output by default

AUDIO_PADDING = 0  # padding value for audio vectors to make them of equal length


def load_wav_file(filename):
    """
    Credits: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/input_data.py
    Loads an audio file and returns a float PCM-encoded array of samples.

    Args:
      filename: Path to the .wav file to load.

    Returns:
      Numpy array holding the 90_10_split_from_train_sample data as floats between -1.0 and 1.0.
    """
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        return sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def vectorize_wav_folder(inp_folder_path, skip_folders=("_background_noise_",)):
    """
    Transforms .wav files to numpy vectors
    Since all .wav files may not be of the same length, smaller vectors are right padded with zeros
    X.shape = (# .wav files, largest audio vector)
    y.shape = (# .wav files, )
    classes.shape = (# number of classes, )

    Note: This function requires huge memory as we load the entire dataset into memory and process it
    :param inp_folder_path: path to .wav file data. The folder structure should be as explained in the Organization section in https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data
    :param skip_folders: folders to skip while looking for .wav files in the input folder
    :return: numpy vectors X, Y, classes
    """
    x_raw_audio = []
    y_raw = []
    classes = []

    def audio_folder_gen():
        """
        Fetches valid audio folders
        :return: a generator which returns reference to a folder
        """
        for folder in os.scandir(inp_folder_path):
            if folder.name in skip_folders or not folder.is_dir():
                continue
            yield folder

    # collect all classes
    for token_folder in audio_folder_gen():
        classes.append(token_folder.name)

    # Assumption: all classes are unique phrases
    classes_map = dict(zip(classes, range(len(classes))))  # assigns numbers to each class string

    # collect all audio and vectorize
    for token_folder in audio_folder_gen():
        for audio_file in os.scandir(token_folder.path):
            x_raw_audio.append(load_wav_file(os.path.abspath(audio_file.path)))
            y_raw.append(classes_map[token_folder.name])
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

    Y = np.array(y_raw)  # to binary strings to persist on disk

    classes = np.array(list(classes_map.items()), dtype='|S9')  # to binary strings to persist on disk

    return X, Y, classes


def fetch_features_for_sample(audio_file_path, preemphasis_alpha=0.97, frame_len_in_secs=0.02, frame_step_in_secs=0.01,
                              NFFT=512, nFilts=40, n_leftFrames=23, n_rightFrames=8, max_nframes=98, nclasses=30):
    return extract_mel_filter_bank_features(audio_file_path, preemphasis_alpha, frame_len_in_secs, frame_step_in_secs,
                                            NFFT, nFilts, n_leftFrames, n_rightFrames, max_nframes)


def get_features_from_all_files(input_folder_path, skip_folders=("_background_noise_",)):
    classes = []
    X = []
    y = []
    for token_folder in os.scandir(input_folder_path):
        if token_folder.name in skip_folders or not token_folder.is_dir():
            continue
        classes.append(token_folder.name)
        for audio_file in os.scandir(token_folder.path):
            audio_file = os.path.abspath(audio_file.path)
            if not audio_file.endswith('.wav'):
                continue
            X.append(fetch_features_for_sample(audio_file, n_leftFrames=0, n_rightFrames=0))
            y.append(token_folder.name)
        print("%s: Completed processing %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), token_folder.name))
    return X, y, classes


def save_train_data(X_train, X_test, y_train, y_test, classes, output_folder):
    """
    Persists the numpy vectors to disk
    :param X_train: numpy vector X for training
    :param X_test: numpy vector X for testing
    :param y_train: numpy vector y for training
    :param y_test: numpy vector y for testing
    :param classes: numpy vector having a list of all classes
    :param output_folder: location where to save the vectors
    :return:
    """
    train_data_file = os.path.join(output_folder, 'train_sounds.h5')
    test_data_file = os.path.join(output_folder, 'test_sounds.h5')
    classes_data_file = os.path.join(output_folder, 'classes_sounds.h5')
    with h5py.File(train_data_file, 'w') as hf:
        hf.create_dataset("train_set_x", data=X_train)
        hf.create_dataset("train_set_y", data=y_train)
    with h5py.File(test_data_file, 'w') as hf:
        hf.create_dataset("test_set_x", data=X_test)
        hf.create_dataset("test_set_y", data=y_test)
    with h5py.File(classes_data_file, 'w') as hf:
        hf.create_dataset("classes", data=classes)
    print("Saved the data to %s, %s and %s" % (train_data_file, test_data_file, classes_data_file))


def save_eval_data(X, output_folder):
    eval_data_file = os.path.join(output_folder, 'eval_sounds.h5')
    with h5py.File(eval_data_file, 'w') as hf:
        hf.create_dataset("eval_set_x", data=X)

    print("Saved the data to %s" % (eval_data_file,))


def load_data(train_data_file="../data/vectorized/90_10_split_from_train_sample/train_sounds.h5",
              test_data_file="../data/vectorized/90_10_split_from_train_sample/test_sounds.h5",
              classes_data_file="../data/vectorized/90_10_split_from_train_sample/classes_sounds.h5"):
    """
    Load vectors from disk and do initial pre-processing
    :param train_data_file: location of train h5 file
    :param test_data_file: location of test h5 file
    :param classes_data_file: location of classes h5 file
    :return: numpy vectors X_train, Y_train, X_test, Y_test, classes
    """
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


def load_data_mfcc(train_data_file="../data/vectorized/90_10_split_from_train_sample/train_sounds.h5",
                   test_data_file="../data/vectorized/90_10_split_from_train_sample/test_sounds.h5",
                   classes_data_file="../data/vectorized/90_10_split_from_train_sample/classes_sounds.h5"):
    with h5py.File(train_data_file, 'r') as hf:
        x_train_orig = hf["train_set_x"][:]
        y_train_orig = hf["train_set_y"][:]
    with h5py.File(test_data_file, 'r') as hf:
        x_test_orig = hf["test_set_x"][:]
        y_test_orig = hf["test_set_y"][:]
    with h5py.File(classes_data_file, 'r') as hf:
        classes = hf["classes"][:]

    # re-shape to perform convolution operations

    x_train_orig = x_train_orig.reshape(x_train_orig.shape[0], x_train_orig.shape[1], x_train_orig.shape[2], 1)
    x_test_orig = x_test_orig.reshape(x_test_orig.shape[0], x_test_orig.shape[1], x_test_orig.shape[2], 1)
    y_train_orig = y_train_orig.reshape(y_train_orig.shape[0], 1)
    y_test_orig = y_test_orig.reshape(y_test_orig.shape[0], 1)

    return x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes


def convert_strings_to_one_hot(Y, classes):
    """
    Converts numpy vector Y of type strings to one hot encoding for training
    :param Y: numpy Y vector of shape (class string, 1)
    :param classes: list of all possible classes
    :return: numpy vector of shape (Y.shape[0], len(classes))
    """
    return label_binarize(Y, classes=classes)


def convert_to_one_hot(Y, classes):
    """
    converts a 1 dimensional numpy array of length n, to n dimensional 2-D numpy array
    :param Y: numpy array to encode
    :param classes: list of all possible values in Y
    :return: one hot encoded 2-D array
    """
    C = len(classes)
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


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


def vectorized_evaluation_chunks(eval_folder, chunk_size):
    """
    Vectorizes evaluation (unlabelled) data and returns in chunks. Hence does not require entire evaluation dataset to
    fit in memory
    :param eval_folder: path to folder where files to evaluate are present
    :param chunk_size: number of evaluation files to return at once
    :return: chunks of evaluation data via a generator
    """

    def pad_audio(x_raw_audio):
        df = pd.DataFrame(x_raw_audio, dtype=float).fillna(AUDIO_PADDING)
        return np.array(df)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    x_raw_audio = []
    source_files = []
    current_batch_size = 0
    for audio_file in os.scandir(eval_folder):
        if not audio_file.name.endswith(".wav"):
            continue
        x_raw_audio.append(load_wav_file(os.path.abspath(audio_file.path)))
        source_files.append(audio_file.name)
        current_batch_size += 1

        if current_batch_size >= chunk_size:
            yield pad_audio(x_raw_audio), source_files

            current_batch_size = 0
            x_raw_audio = []
            source_files = []

    if len(x_raw_audio) > 0:
        yield pad_audio(x_raw_audio), source_files


def extract_mel_filter_bank_features(wavfile, preemphasis_alpha, frame_len_in_secs, frame_step_in_secs, NFFT, nFilts,
                                     n_leftFrames, n_rightFrames, max_nframes):
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
    count = 0
    for i in np.arange(n_leftFrames, nFrames - n_rightFrames):
        filter_bank_features_normalized_final[count:] = filter_bank_features_normalized[
                                                        i - n_leftFrames:i + n_rightFrames + 1, :].flatten()
        count += 1

    return filter_bank_features_normalized_final


def prepare_sample_from_vector(h5_inp_folder, output_folder, train_sample_size=1000, test_sample_size=60,
                               data_loader=load_data):
    """
    Samples a h5py file generated from a audio vector train data set
    :param h5_inp_folder: folder containing train_sounds, test_sounds and classes_sounds vectorized h5 files
    :param output_folder: directory where to save the sampled files
    :param train_sample_size: number of examples as training data in the output
    :param test_sample_size: number of examples as test data in the output
    :param data_loader: a function handler which can load previously saved vectorized data
    :return:
    """
    train_data_file = os.path.join(h5_inp_folder, 'train_sounds.h5')
    test_data_file = os.path.join(h5_inp_folder, 'test_sounds.h5')
    classes_data_file = os.path.join(h5_inp_folder, 'classes_sounds.h5')
    X, Y, _, _, classes = data_loader(train_data_file, test_data_file, classes_data_file)
    print("%s: Vectorized training data" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))
    X_mini, y_mini = next(random_mini_batches(X, Y, train_sample_size + test_sample_size))  # fetch the first batch
    test_size = float(test_sample_size) / (test_sample_size + train_sample_size)
    X_train, X_test, y_train, y_test = train_test_split(X_mini, y_mini, test_size=test_size, random_state=42)
    save_train_data(X_train, X_test, y_train, y_test, classes, output_folder)
    print("%s: Saved the sample" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))


def prepare_mfcc_sample_from_wav(input_dir, output_dir, percent_to_sample=20, test_train_ratio=0.1, words=None):
    model_settings = {
        "desired_samples": 160,
        "fingerprint_size": 40,
        "label_count": 4,
        "window_size_samples": 100,
        "window_stride_samples": 100,
        "dct_coefficient_count": 40,
    }
    classes = set()
    X = []
    y = []
    if not words:
        words = ['bird', 'on', 'cat', 'bed', 'wow', 'dog', 'five', 'right', 'zero', 'two', 'yes', 'stop', 'tree',
                 'left', 'one', 'off', 'down', 'nine', 'house', 'marvin', 'eight', 'six', 'go', 'three', 'four', 'no',
                 'happy', 'seven', 'up', 'sheila']
    # using a helper code to sample files from the input folder
    audio_processor = AudioProcessor("", input_dir, 0, 0, words,
                                     0, 100 - percent_to_sample, model_settings)
    print("%s: Selected %d samples from given dataset" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                          len(audio_processor.data_index["training"])))
    for sample in audio_processor.data_index["training"]:
        X.append(fetch_features_for_sample(sample["file"]))
        y.append(sample["label"])
        classes.add(sample["label"])
    print("%s: Created the sample" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))

    # Other helpful functions which not only sample but also add background noise and return X, y
    # Not using these directly as they do feature extraction for X than what we want
    # with tf.Session as sess:
    #     result_data, result_labels = audio_processor.get_unprocessed_data(10, model_settings, "training")
    #     result_data, result_labels = audio_processor.get_data(10, 0, model_settings, 0.3, 0.1, 100, "training", sess)

    y = np.array(y, dtype='|S9')
    classes = np.array(list(classes), dtype='|S9')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_train_ratio, random_state=42)
    save_train_data(X_train, X_test, y_train, y_test, classes, output_dir)
    print("%s: Saved the sample" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))

    return X, y, classes


def main():
    # X, Y, classes = vectorize_wav_folder("../data/train/audio")
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    # save_train_data(X_train, X_test, y_train, y_test, classes, "../data/vectorized/90_10_split_from_train2/")
    # x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes = load_data()

    # input_folder_path = '../data/train/audio/'
    # X, Y, classes = get_features_from_all_files(input_folder_path, skip_folders=("_background_noise_",))
    # Y = np.array(Y, dtype='|S9')  # to binary strings to persist on disk
    # classes = np.array(classes, dtype='|S9')  # to binary strings to persist on disk
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    # print("Finished splitting all data to train-test")
    # save_train_data(X_train, X_test, y_train, y_test, classes, "../data/vectorized/mfcc_zero_context/")

    # prepare_sample_from_vector("../data/vectorized/90_10_split_from_train2/",
    #                            "../data/vectorized/90_10_split_from_train2_sample/")

    # prepare_mfcc_sample_from_wav("../data/train/audio/", "../data/vectorized/mfcc_20percent-0.1test_sample/", 20)

    # X = vectorize_evaluation_wav("../data/test/audio/")
    print("%s: Completed vectorizing test data" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))
    # save_eval_data(X, "../data/test/vectorized/")


if __name__ == '__main__':
    main()
