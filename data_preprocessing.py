import librosa
import numpy as np
import time
import pickle
import scipy
from scipy import signal


def replicate_elements_in_array(arr, num_reps):
    return np.array([x for x in arr for _ in [r for r in range(num_reps)]])


def read_all_music_benchmark(base_path, classifications):
    """
    Read the songs in the Benchmark dataset that fall into the given classifications.
    There are 120 (blues) + 300 (hip-pop) + 319 (jazz) + 116 (pop) + 504 (rock) = 1359 songs.
    Each song is around 10 seconds long and the sample rate is 44100 Hz.
    The samples are resampled to be 22050 Hz.
    :param base_path: path to the dataset.
    :param classifications: the classifications to load.
    :return: the music data, the labels for each music.
    """
    print('Loading files for the Benchmark dataset.')
    benchmark_sample_rate = 44100
    expected_music_length = 10
    return read_all_music(base_path, classifications, benchmark_sample_rate)


def read_all_music_gtzan(base_path, classifications):
    """
    Read the songs in the GTZAN dataset that fall into the given classifications.
    There are 100 * 5 = 500 songs.
    Each song is around 30 seconds long and the sample rate is 22050 Hz.
    :param base_path: path to the GTZAN dataset.
    :param classifications: the classifications to load.
    :return: the music data, the labels for each music.
    """
    print('Loading files for the GTZAN dataset.')
    gtzan_sample_rate = 22050
    expected_music_length = 30
    return read_all_music(base_path, classifications, gtzan_sample_rate)


def read_all_music(base_path, classifications, sample_rate):
    """
    Read the songs in the dataset specified by the base path and which fall into the given classifications.
    Some samples are cut to be a little shorter to ensure that the spectrograms have the same dimensions.
    :param base_path: path to the dataset.
    :param classifications:  the classifications to load.
    :param sample_rate: sample rate of the songs.
    :return: the music data, the labels for each music.
    """
    dataset = []
    labels = []

    # todo: remove debugging code
    max_length = -1
    min_length = 100000000

    counter = 0
    for i in range(len(classifications)):
        classification = classifications[i]
        path_to_classification = base_path + classification
        print('Loading files under', path_to_classification)
        files = librosa.util.find_files(path_to_classification, ext=['mp3', 'wav'])
        files = np.asarray(files)

        for path_to_music in files:
            x, sample_rate = librosa.load(path_to_music, sr=sample_rate)
            if sample_rate != 22050:
                # resample the sample if the sample rate is not 22050 Hz.
                x = librosa.resample(x, sample_rate, 22050)
            len_x = len(x)

            # todo: remove debugging code
            if len_x > max_length:
                max_length = len_x
                print('warning 1: max_length is', max_length, 'at', path_to_music)
            if len_x < min_length:
                min_length = len_x
                print('warning 2: min_length is', min_length, 'at', path_to_music)

            dataset.append(x)
            labels.append(i)
            counter += 1
    try:
        dataset = np.array(dataset)
    except:
        print('Warning 10: cannot convert dataset to numpy array.')
    return dataset, np.array(labels)


def spectrograms_benchmark(dataset, labels, segment_length, window_length, overlap_ratio):
    """
    Compute the spectrograms for the songs in the Benchmark dataset.
    The data are resampled from 44100 Hz to 22050 Hz.
    :param dataset: relevant data in the Benchmark dataset.
    :param labels: labels for the Benchmark dataset.
    :param segment_length: length of the music segment in seconds passed to create spectrogram.
    :param window_length: the number of points in the output window.
    :param overlap_ratio: The overlapping ratio of the windows.
    :return: the spectrograms.
    """
    expected_music_length = 10
    return spectrogram_of_dataset(dataset, labels, expected_music_length, segment_length,
                                  window_length, overlap_ratio)


def spectrograms_gtzan(dataset, labels, segment_length, window_length, overlap_ratio):
    """
    Compute the spectrograms for the songs in the GTZAN dataset.
    Each sample is partitioned into approximately 3 equal parts since each song was originally 30 seconds long
    and thus we analyze samples corresponding to 10-second long clips.
    :param dataset: relevant data in the GTZAN dataset.
    :param labels: labels for the GTZAN dataset.
    :param segment_length: length of the music segment in seconds passed to create spectrogram.
    :param window_length: the number of points in the output window.
    :param overlap_ratio: The overlapping ratio of the windows.
    :return: the spectrograms.
    """
    expected_music_length = 30
    return spectrogram_of_dataset(dataset, labels, expected_music_length,
                                  segment_length, window_length, overlap_ratio)


def spectrogram_of_dataset(dataset, labels, expected_music_length, segment_length, window_length, overlap_ratio):
    """
    Compute the spectrograms for the samples in the given dataset.
    :param dataset: the dataset containing the samples.
    :param expected_music_length: the time length in seconds for the original music sample.
    :param segment_length: length of the music segment in seconds passed to create spectrogram.
    :param window_length: the number of points in the output window.
    :param overlap_ratio: The overlapping ratio.
    :return: the spectrograms, labels for each spectrogram
    """
    if expected_music_length not in (10, 30):
        raise Exception('Variable expected_music_length should be either 10 or 30, but received {}.'
                        .format(expected_music_length))

    spectrograms = []
    spectrogram_labels = []

    window = scipy.signal.hanning(window_length)
    noverlap = window_length // (1/overlap_ratio)


    # (len(x) - window_length) // (window_length - window_length//(1/overlap_ratio)) + 1 = spectrogram number of columns
    # For testing:
    # window_length_to_max_sample_length = {256: 999999, 512: 999999, 1024: 999999}
    # 3 seconds:
    # window_length_to_max_sample_length = {256: 66559, 512: 66559, 1024: 66559}
    # 5 seconds:
    # window_length_to_max_sample_length = {256: 110079, 512: 110079, 1024: 110079}
    # 10 seconds
    # window_length_to_max_sample_length = {512: 220159, 1024: 220159, 2048: 220159}

    # upper_bound = window_length_to_max_sample_length[window_length]

    # upper_bound = 999999
    # if segment_length == 3:
    #     # length will be 127
    #     upper_bound = 66047
    # elif segment_length == 5:
    #     upper_bound = 110079
    # elif segment_length == 10:
    #     upper_bound = 220159

    # limit the lower and upper bounds to specify the spectrogram lengths.
    lower_bound = 0
    if segment_length == 3:
        # length will be 128
        lower_bound = 66560
    elif segment_length == 5:
        lower_bound = 110080
    elif segment_length == 10:
        lower_bound = 220160

    upper_bound = 999999
    if segment_length == 3:
        # length will be 128
        upper_bound = 66559
    elif segment_length == 5:
        upper_bound = 999999
    elif segment_length == 10:
        upper_bound = 999999

    max_spec_length = -1
    min_spec_length = 1000000000
    for i in range(len(dataset)):
        sample = np.array(dataset[i])

        sample_partitions = get_spectrogram_partitions(sample, expected_music_length, segment_length, lower_bound, upper_bound)
        # length of each partition

        # if sample_partitions.shape[1] > upper_bound:
        #     sample_partitions = sample_partitions[:, 0:upper_bound]


        # if expected_music_length == 10 and len_x > upper_bound:
        #     # need to ensure the length is less than upper_bound to have spectrograms of the same dimension
        #     sample_partitions = sample_partitions[:, 0:upper_bound]
        # elif expected_music_length == 30 and len_x > upper_bound * 3:
        #     # need to ensure the length is less than upper_bound * 3 to have spectrograms of the same dimension
        #     sample_partitions = sample_partitions[:, 0:upper_bound * 3]

        for j in range(len(sample_partitions)):
            spectrogram_labels.append(labels[i])

            x = sample_partitions[j]

            # if expected_music_length == 10:
            frequencies, times, Sxx = signal.spectrogram(x, window=window, noverlap=noverlap)
            spectrograms.append(Sxx)

            # todo: remove debugging code
            if len(Sxx[0]) > max_spec_length:
                max_spec_length = len(Sxx[0])
                print('warning 3: max_spec_length is', max_spec_length, 'with len(x)', len(x))
            if len(Sxx[0]) < min_spec_length:
                min_spec_length = len(Sxx[0])
                print('warning 4: min_spec_length is', min_spec_length, 'with len(x)', len(x))

            # else:
            #     # expected_music_length == 30
            #     # partition the partitioned sample into 3 equal parts if the song was 30 seconds long originally.
            #     partition_size = len(x) // 3
            #     frequencies_1, times_1, Sxx_1 = signal.spectrogram(x[:partition_size], window=window, noverlap=noverlap)
            #     frequencies_2, times_2, Sxx_2 = signal.spectrogram(x[partition_size:partition_size * 2], window=window, noverlap=noverlap)
            #     frequencies_3, times_3, Sxx_3 = signal.spectrogram(x[partition_size * 2:], window=window, noverlap=noverlap)
            #
            #     # todo: remove debugging code
            #     if len(Sxx_1[0]) > max_spec_length:
            #         max_spec_length = len(Sxx_1[0])
            #         print('warning 5: max_spec_length is', max_spec_length, 'with len(x)', len(x))
            #     if len(Sxx_1[0]) < min_spec_length:
            #         min_spec_length = len(Sxx_1[0])
            #         print('warning 6: min_spec_length is', min_spec_length, 'with len(x)', len(x))
            #     if len(Sxx_3[0]) > max_spec_length:
            #         max_spec_length = len(Sxx_3[0])
            #         print('warning 7: max_spec_length is', max_spec_length, 'with len(x)', len(x))
            #     if len(Sxx_3[0]) < min_spec_length:
            #         min_spec_length = len(Sxx_3[0])
            #         print('warning 8: min_spec_length is', min_spec_length, 'with len(x)', len(x))
            #
            #     spectrograms.append(Sxx_1)
            #     spectrograms.append(Sxx_2)
            #     spectrograms.append(Sxx_3)

    try:
        spectrograms = np.array(spectrograms)
    except:
        # cannot convert list of spectrograms to numpy array if the spectrograms have different dimensions.
        print('Warning 11: cannot convert spectrograms to numpy array.')
    try:
        spectrogram_labels = np.array(spectrogram_labels)
    except:
        print('Warning 12: cannot convert spectrograms labels to numpy array.')
    return spectrograms, spectrogram_labels


def get_spectrogram_partitions(sample, expected_music_length, segment_length, lower_bound, upper_bound):
    """
    Partition the sample into specified lengths. The extra end is truncated.
    :param sample: a sample.
    :param expected_music_length: the time length in seconds for the original music sample.
    :param segment_length: length of the music segment in seconds passed to create spectrogram.
    :param lower_bound: lower bound of length for partitioned sample.
    :param upper_bound: upper bound of length for partitioned sample.
    :return: partitions of the sample.
    """

    num_partitions = expected_music_length // segment_length
    len_sample = len(sample)
    len_partition = len_sample // num_partitions

    if len_partition < lower_bound:
        len_partition = upper_bound
        num_partitions = num_partitions - 1
        # print('Warning 9: length of partition less than lower bound. Set to upper bound.')
    if len_partition > upper_bound:
        len_partition = upper_bound
        # print('Warning 10: length of partition larger than upper bound. Set to upper bound.')

    sample_partitions = np.empty((num_partitions, len_partition))

    for i in range(num_partitions):
        start_index = len_partition * i
        sample_partitions[i, :] = sample[start_index: start_index + len_partition]
    return sample_partitions


def data_preprocessing():
    """
    Preprocess data, includes loading the Benchmark dataset and the GTZAN dataset,
    then compute the spectrograms for each data.
    """
    print('Preprocessing data...')

    load_original_datasets = False
    if load_original_datasets:

        classifications = np.array(['blues', 'hiphop', 'jazz', 'pop', 'rock'], dtype=object)
        # classifications = np.array(['blues'], dtype=object)
        # todo: remove debugging code above

        # paths to datasets
        benchmark_base_path = 'dataset/benchmarkdataset/'
        gtzan_base_path = 'dataset/gtzan/'

        # read music files
        t_1 = time.time()
        benchmark_dataset, benchmark_labels = read_all_music_benchmark(benchmark_base_path, classifications)

        pickle.dump(benchmark_dataset, open('dataset/benchmark_dataset.p', 'wb'))
        pickle.dump(benchmark_labels, open('dataset/benchmark_labels_not_replicated.p', 'wb'))


        t_2 = time.time()
        gtzan_dataset, gtzan_labels = read_all_music_gtzan(gtzan_base_path, classifications)


        pickle.dump(gtzan_dataset, open('dataset/gtzan_dataset.p', 'wb'))
        pickle.dump(gtzan_labels, open('dataset/gtzan_labels_not_replicated.p', 'wb'))

        t_3 = time.time()
        print('The time used for loading the Benchmark dataset was', t_2 - t_1)
        print('The time used for loading the GTZAN dataset was', t_3 - t_2)
        print('The total time used for loading datasets was', t_3 - t_1)
        print()

    # the length of music segments in seconds passed in to create spectrograms. Should be in (0, 10).
    segment_length = 3
    # length of window, should be a power of 2 for faster computation.
    window_length = 1024
    # should use 50% window overlapping percentage.
    overlap_ratio = 1 / 2

    benchmark_dataset = pickle.load(open('dataset/benchmark_dataset.p', 'rb'))
    gtzan_dataset = pickle.load(open('dataset/gtzan_dataset.p', 'rb'))

    benchmark_labels = pickle.load(open('dataset/benchmark_labels_not_replicated.p', 'rb'))
    gtzan_labels = pickle.load(open('dataset/gtzan_labels_not_replicated.p', 'rb'))

    # benchmark_labels = replicate_elements_in_array(benchmark_labels, 10 // segment_length)
    # gtzan_labels = replicate_elements_in_array(gtzan_labels, 30 // segment_length)

    # compute spectrograms
    t_4 = time.time()
    benchmark_spectrograms, benchmark_labels = spectrograms_benchmark(benchmark_dataset, benchmark_labels,
                                                                      segment_length, window_length, overlap_ratio)
    t_5 = time.time()
    print('The time used for calculating spectrograms for the Benchmark dataset was', t_5 - t_4)
    gtzan_spectrograms, gtzan_labels = spectrograms_gtzan(gtzan_dataset, gtzan_labels,
                                                          segment_length, window_length, overlap_ratio)
    t_6 = time.time()
    print('The time used for calculating spectrograms for the GTZAN dataset was', t_6 - t_5)

    print('Window length is', window_length, 'overlap ratio is', overlap_ratio,
          'seg length is', segment_length)
    print()

    # pickling data
    # modify seg_length_suffix to choose dataset
    seg_length_suffix = '_' + str(segment_length)
    window_length_suffix = '_' + str(window_length)
    pickle.dump(benchmark_spectrograms,
                open('dataset/benchmark_spectrograms' + seg_length_suffix + window_length_suffix + '.p', 'wb'))
    pickle.dump(gtzan_spectrograms,
                open('dataset/gtzan_spectrograms' + seg_length_suffix + window_length_suffix + '.p', 'wb'))
    pickle.dump(benchmark_labels, open('dataset/benchmark_labels' + seg_length_suffix + window_length_suffix + '.p', 'wb'))
    pickle.dump(gtzan_labels, open('dataset/gtzan_labels' + seg_length_suffix + window_length_suffix + '.p', 'wb'))
