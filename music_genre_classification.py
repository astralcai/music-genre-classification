import sys
import time
import pickle

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
    benchmark_spectrograms = pickle.load(open('benchmark_spectrograms.p', 'rb'))
    gtzan_spectrograms = pickle.load(open('gtzan_spectrograms.p', 'rb'))
    benchmark_labels = pickle.load(open('benchmark_labels.p', 'rb'))
    gtzan_labels = pickle.load(open('gtzan_labels_labels.p', 'rb'))

    print(benchmark_spectrograms.shape)
    print(gtzan_spectrograms.shape)
    print(benchmark_labels.shape)
    print(gtzan_labels.shape)


main()
