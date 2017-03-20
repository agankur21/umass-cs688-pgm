import numpy as np

from config.config import *


def parse_model_params(file_path=MODEL_PATH_FEATURES):
    """
    Parse the params from the model file in the form of a text file
    :return:
    """
    params = None
    try:
        params = np.loadtxt(file_path, delimiter=" ")
    except Exception as e:
        print e.message
    finally:
        return params


def parse_features(file_path):
    """
    Parse the features from a file such that each line is a new character
    :return:
    """
    features = None
    try:
        features = np.loadtxt(file_path, delimiter=" ")
    except Exception as e:
        print e.message
    finally:
        return features


def parse_true_labels(word_index, data_type='test'):
    """
    Get the true label indices for the words given
    :param word_index:
    :return:
    """
    file_path = os.path.join(DATA_FOLDER, "%s_words.txt" % data_type)
    all_words = open(file_path, 'r').read().splitlines()
    word = list(all_words[word_index - 1])
    label_array = np.zeros(len(word))
    for i in range(len(word)):
        label_array[i] = ENCODING_KEY[word[i]]
    return "".join(word), label_array


def label_encodings(word_list):
    label_array = np.zeros(len(word_list))
    for i in range(len(word_list)):
        label_array[i] = ENCODING_KEY[word_list[i]]
    return label_array


def get_all_possible_label_sequences(sequence_length, temp_list, out):
    if len(temp_list) == sequence_length:
        out.append(list(temp_list))
        return
    else:
        for i in range(len(DECODING_KEY)):
            temp_list.append(i)
            get_all_possible_label_sequences(sequence_length, temp_list, out)
            temp_list.pop()


def get_marginal_sequences(index,value,num_features):
    rest_label_sequence=[]
    get_all_possible_label_sequences(num_features-1,[],rest_label_sequence)
    out=[]
    for label_sequence in rest_label_sequence:
        out.append(label_sequence[0:index]+[value]+label_sequence[index:])
    return out

