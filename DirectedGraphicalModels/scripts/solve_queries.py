import os
import re

import numpy as np

from entity.Data import Data
from entity.MetaData import MetaData
from entity.Parameters import Parameters

num_files = 5
parent_info_dict={
    'A' : [],
    'G' : [],
    'BP': ['G'],
    'CH': ['A','G'],
    'HD':['BP', 'CH'],
    'CP': ['HD'],
    'EIA': ['HD'],
    'ECG': ['HD'],
    'HR': ['HD','A']
}


def print_prob_dict(prob_map, level=0):
    for key, prob in prob_map.iteritems():
        if level == 0:
            print "%s,%0.4f" % (key, prob)
        else:
            for key2, value in prob.iteritems():
                key = re.sub(r'[\(\) ]', "", str(key))
                print "%s,%s,%0.4f" % (key2, key, value)


def solve_q4(meta_data_path, training_data_path):
    meta_data = MetaData(meta_data_path)
    data = Data(meta_data)
    data.populate_data(training_data_path)
    parameters = Parameters(data,parent_info_dict)
    print "Printing Probability distribution for P(A)"
    print_prob_dict(parameters.probability_a)
    # P(BP/G)
    print "Printing Probability distribution for P(BP/G)"
    print_prob_dict(parameters.probability_bp, level=1)
    # P(HD|BP,CH)
    print "Printing Probability distribution for P(HD|BP,CH)"
    print_prob_dict(parameters.probability_hd, level=1)
    # P(HR|A,HD)
    print "Printing Probability distribution for P(HR|HD,A)"
    print_prob_dict(parameters.probability_hr, level=1)

    return parameters


def solve_q5(trained_parameters):
    # P(CH|A,G,CP,BP,ECG,HR,EIA,HD)
    print "Printing Probability distribution for P(CH|A=2,G=M,CP=None,BP=L,ECG=Normal,HR=L,EIA =No,HD=No)"
    value_dict1 = {'CH': 1, 'A': 2, 'G': 2, 'CP': 4, 'BP': 1, 'ECG': 1, 'HR': 1, 'EIA': 1, 'HD': 1}
    value_dict2 = {'CH': 2, 'A': 2, 'G': 2, 'CP': 4, 'BP': 1, 'ECG': 1, 'HR': 1, 'EIA': 1, 'HD': 1}
    joints_probabilities = map(lambda x: trained_parameters.get_point_probability(x), [value_dict1, value_dict2])
    print_prob_dict({1: joints_probabilities[0] / sum(joints_probabilities),
                     2: joints_probabilities[1] / sum(joints_probabilities)})
    print "Printing Probability distribution for P(BP|A=2,CP=1,CH=2,ECG=1,HR=2,EIA=2,HD=1))"
    value_dict1 = {'BP': 1, 'G': 1, 'A': 2, 'CP': 1, 'CH': 2, 'ECG': 1, 'HR': 2, 'EIA': 2, 'HD': 1}
    value_dict2 = {'BP': 1, 'G': 2, 'A': 2, 'CP': 1, 'CH': 2, 'ECG': 1, 'HR': 2, 'EIA': 2, 'HD': 1}
    value_dict3 = {'BP': 2, 'G': 1, 'A': 2, 'CP': 1, 'CH': 2, 'ECG': 1, 'HR': 2, 'EIA': 2, 'HD': 1}
    value_dict4 = {'BP': 2, 'G': 2, 'A': 2, 'CP': 1, 'CH': 2, 'ECG': 1, 'HR': 2, 'EIA': 2, 'HD': 1}
    joints_probabilities = map(lambda x: trained_parameters.get_point_probability(x),
                               [value_dict1, value_dict2, value_dict3, value_dict4])
    print_prob_dict({1: (joints_probabilities[0] + joints_probabilities[1]) / sum(joints_probabilities),
                     2: (joints_probabilities[2] + joints_probabilities[3]) / sum(joints_probabilities)})


def train_data(meta_data_path, training_data_folder):
    """
    Q6(a). Train different parameters for each training File
    Q6(b). Printing the probability distribution of HD|A,BP,CH,CP,EIA,ECG,HR for all training distributions
    :param meta_data_path:
    :param training_data_folder:
    :return:
    """
    meta_data = MetaData(meta_data_path)
    list_train_parameters = []
    for i in range(1, num_files + 1):
        train_file_name = os.path.join(training_data_folder, "data-train-%d.txt" % i)
        data = Data(meta_data)
        data.populate_data(train_file_name)
        params = Parameters(data,parent_info_dict)
        list_train_parameters.append(params)
    return list_train_parameters


def cross_validation(meta_data_path, list_train_parameters, test_data_folder):
    """
    Q6(c): Calculating and Reporting the mean accuracy and standard deviation of tne accuracy
    :param meta_data_path:
    :param list_train_parameters:
    :param test_data_folder:
    :return:
    """
    meta_data = MetaData(meta_data_path)
    list_prediction_accuracy = []
    for i in range(len(list_train_parameters)):
        print "Testing data for Test File number %d" % (i + 1)
        train_params = list_train_parameters[i]
        test_data = Data(meta_data)
        test_data.populate_data(os.path.join(test_data_folder, "data-test-%d.txt" % (i + 1)))
        actual_hd = test_data.data['HD']
        predicted_hd = test_data.data.apply(lambda x: train_params.predict(x, 'HD'), axis=1)
        prediction_accuracy = (predicted_hd == actual_hd).sum() * 1.0 / actual_hd.count()
        print "Prediction Accuracy on data-test-%d.txt : %2.4f%s" %(i+1,prediction_accuracy*100,'%')
        list_prediction_accuracy.append(prediction_accuracy)
    print "Mean of prediction accuracy: %2.4f%s" % (np.mean(list_prediction_accuracy) * 100, '%')
    print "Std. Deviation of  prediction accuracy: %2.4f%s" % (np.std(list_prediction_accuracy) * 100, '%')


if __name__ == '__main__':
    meta_data_path = os.path.join(os.getcwd(), "../../Data/data_info.txt")
    training_data_path1 = os.path.join(os.getcwd(), "../../Data/data-train-1.txt")
    print "Printing solutions for Q4"
    #trained_params = solve_q4(meta_data_path, training_data_path1)
    print "Printing solutions for Q5"
    #solve_q5(trained_params)
    print "Printing solutions for Q6"
    list_train_parameters = train_data(meta_data_path, os.path.join(os.getcwd(), "../../Data"))
    cross_validation(meta_data_path, list_train_parameters, os.path.join(os.getcwd(), "../../Data"))
