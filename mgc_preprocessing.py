import sys
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np

import audio_preprocessing


def main():
    run_data_preprocessing = False
    try:
        # check if an argument is provided
        sys.argv[1]
    except:
        pass
    else:
        if sys.argv[1].lower() == 'true':
            # if an argument with indicating a boolean True is passed in, then
            # run data preprocessing (load datasets and calculate spectrograms)
            run_data_preprocessing = True

    if run_data_preprocessing:
        st_data_preprocessing = time.time()
        audio_preprocessing.data_preprocessing()
        print('The total time used for loading and preprocessing the data was', time.time() - st_data_preprocessing)

    # modify the two parameters below to choose dataset
    # segment length in seconds passed to generate spectrograms, and window length.
    segment_length = 3
    window_length = 1024

    # load spectrograms
    seg_length_suffix = '_' + str(segment_length)
    window_length_suffix = '_' + str(window_length)
    benchmark_spectrograms = pickle.load(
        open('dataset/benchmark_spectrograms' + seg_length_suffix + window_length_suffix + '.p', 'rb'))
    gtzan_spectrograms = pickle.load(
        open('dataset/gtzan_spectrograms' + seg_length_suffix + window_length_suffix + '.p', 'rb'))
    benchmark_labels = pickle.load(open('dataset/benchmark_labels' + seg_length_suffix + window_length_suffix + '.p', 'rb'))
    gtzan_labels = pickle.load(open('dataset/gtzan_labels' + seg_length_suffix + window_length_suffix + '.p', 'rb'))

    print('Spectrograms loaded.')
    print('The dimension of the dataset for spectrograms from the Benchmark dataset is')
    print(benchmark_spectrograms.shape)
    print('The dimension of the dataset for spectrograms from the Tzanetakis\' dataset is')
    print(gtzan_spectrograms.shape)
    print('The number of labels for the Benchmark spectrograms is')
    print(benchmark_labels.shape)
    print('The number of labels for the Tzanetakis\' spectrograms is')
    print(gtzan_labels.shape)

    # plot example spectrograms
    spectrogram = benchmark_spectrograms[0]
    plt.pcolormesh(10 * np.log10(spectrogram))
    plt.title('Spectrogram')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.colorbar()
    plt.show()

    spectrogram = gtzan_spectrograms[0]
    plt.pcolormesh(10 * np.log10(spectrogram))
    plt.title('Spectrogram')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.colorbar()
    plt.show()


main()
