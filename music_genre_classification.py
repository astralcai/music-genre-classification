import sys
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np

import data_preprocessing


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
        data_preprocessing.data_preprocessing()
        print('The total time used for loading and preprocessing the data was', time.time() - st_data_preprocessing)

    # Use the following to load spectrograms:
    # modify name_suffix to choose dataset
    name_suffix = '_512'
    benchmark_spectrograms = pickle.load(open('dataset/benchmark_spectrograms' + name_suffix + '.p', 'rb'))
    gtzan_spectrograms = pickle.load(open('dataset/gtzan_spectrograms' + name_suffix + '.p', 'rb'))
    benchmark_labels = pickle.load(open('dataset/benchmark_labels.p', 'rb'))
    gtzan_labels = pickle.load(open('dataset/gtzan_labels_labels.p', 'rb'))

    print(benchmark_spectrograms[0].shape)
    print(benchmark_spectrograms.shape)
    print(gtzan_spectrograms.shape)
    print(benchmark_labels.shape)
    print(gtzan_labels.shape)

    # todo: remove the temporary example for plotting a spectrogram.
    spectrogram = gtzan_spectrograms[5]

    plt.pcolormesh(10 * np.log10(spectrogram))
    plt.colorbar()
    plt.show()


main()
