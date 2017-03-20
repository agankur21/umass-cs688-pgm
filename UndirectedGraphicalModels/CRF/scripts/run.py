from entity.linear_chain_model import linear_chain_model
from utils.crf import *
from utils.input import *
from utils.optimize import *


def solve_q1_1():
    test_file_path = os.path.join(DATA_FOLDER, "test_img1.txt")
    features = parse_features(test_file_path)
    params = parse_model_params(MODEL_PATH_FEATURES)
    log_potential_matrix = CRF.log_sequence_potential(features, params)
    print "------------Solution for Q1.1---------------"
    print "The log potential table for test image 1:"
    print log_potential_matrix


def solve_q1_2():
    features_params = parse_model_params(MODEL_PATH_FEATURES)
    transition_params = parse_model_params(MODEL_PATH_TRANSITION)
    print "------------Solution for Q1.2---------------"
    for i in range(1, 4):
        test_file_path = os.path.join(DATA_FOLDER, "test_img%d.txt" % i)
        features = parse_features(test_file_path)
        word, y = parse_true_labels(i, data_type='test')
        negative_energy = 1.0 * CRF.energy(features, y, features_params, transition_params)
        print "Negative Energy for Test Image of label : '%s' is %g" % (word, negative_energy)


def solve_q1_3():
    features_params = parse_model_params(MODEL_PATH_FEATURES)
    transition_params = parse_model_params(MODEL_PATH_TRANSITION)
    cache_label_sequence = {}
    print "------------Solution for Q1.3---------------"
    for i in range(1, 4):
        test_file_path = os.path.join(DATA_FOLDER, "test_img%d.txt" % i)
        features = parse_features(test_file_path)
        if len(features) not in cache_label_sequence:
            out = []
            get_all_possible_label_sequences(len(features), [], out)
            cache_label_sequence[len(features)] = out
        log_partition_val = CRF.log_partition_naive(features, cache_label_sequence[len(features)], features_params,
                                                    transition_params)
        print "Log Partition for Test Image %d is : %g" % (i, log_partition_val)


def solve_q1_4():
    features_params = parse_model_params(MODEL_PATH_FEATURES)
    transition_params = parse_model_params(MODEL_PATH_TRANSITION)
    cache_label_sequence = {}
    print "------------Solution for Q1.4---------------"
    for i in range(1, 4):
        test_file_path = os.path.join(DATA_FOLDER, "test_img%d.txt" % i)
        features = parse_features(test_file_path)
        if len(features) not in cache_label_sequence:
            out = []
            get_all_possible_label_sequences(len(features), [], out)
            cache_label_sequence[len(features)] = out
        energy_sequence = map(lambda y: CRF.energy(features, y, features_params, transition_params),
                              cache_label_sequence[len(features)])
        max_energy_index = np.argmax(energy_sequence)
        log_partition_val = CRF.log_partition_naive(features, cache_label_sequence[len(features)], features_params,
                                                    transition_params)
        probability = np.exp(energy_sequence[max_energy_index] - log_partition_val)
        char_sequence = map(lambda i: DECODING_KEY[i], cache_label_sequence[len(features)][max_energy_index])
        print (
            "Most probable label for Test Image {} is {} with probability: {}%".format(
                i, "".join(char_sequence), probability * 100))


def solve_q1_5():
    features_params = parse_model_params(MODEL_PATH_FEATURES)
    transition_params = parse_model_params(MODEL_PATH_TRANSITION)
    print "------------Solution for Q1.5---------------"
    test_file_path = os.path.join(DATA_FOLDER, "test_img1.txt")
    features = parse_features(test_file_path)
    marginal_probability_matrix = np.zeros((len(features), len(DECODING_KEY)))
    all_label_sequence = []
    get_all_possible_label_sequences(len(features), [], all_label_sequence)
    partition_val = np.exp(CRF.log_partition_naive(features, all_label_sequence, features_params,
                                                   transition_params))
    for i in range(len(features)):
        for j in range(len(DECODING_KEY)):
            marginal_label_sequence = get_marginal_sequences(i, j, len(features))
            marginal_energy_sequence = map(
                lambda y: np.exp(CRF.energy(features, y, features_params, transition_params)),
                marginal_label_sequence)
            marginal_probability_matrix[i, j] = 1.0 * np.sum(marginal_energy_sequence) / partition_val
    print marginal_probability_matrix


def solve_q_2_1():
    features_params = parse_model_params(MODEL_PATH_FEATURES)
    transition_params = parse_model_params(MODEL_PATH_TRANSITION)
    print "------------Solution for Q2.1---------------"
    test_file_path = os.path.join(DATA_FOLDER, "test_img1.txt")
    features = parse_features(test_file_path)
    model = linear_chain_model(len(features), features_params, transition_params)
    model.x = features
    model.reset_cache()
    # Printing message for m1->2
    log_message_passing_val = map(lambda label: CRF.pair_wise_log_message_passing(0, 1, label, model),
                                  range(len(DECODING_KEY)))
    for i in range(len(DECODING_KEY)):
        print "Log message passing value for m1->2(Y2='%s') = %g" % (DECODING_KEY[i], log_message_passing_val[i])

    # Printing message for m2->1
    log_message_passing_val = map(lambda label: CRF.pair_wise_log_message_passing(1, 0, label, model),
                                  range(len(DECODING_KEY)))
    for i in range(len(DECODING_KEY)):
        print "Log message passing value for m2->1(Y1='%s') = %g" % (DECODING_KEY[i], log_message_passing_val[i])

    # Printing message for m2->3
    log_message_passing_val = map(lambda label: CRF.pair_wise_log_message_passing(1, 2, label, model),
                                  range(len(DECODING_KEY)))
    for i in range(len(DECODING_KEY)):
        print "Log message passing value for m2->3(Y3='%s') = %g" % (DECODING_KEY[i], log_message_passing_val[i])

    # Printing message for m3->2
    log_message_passing_val = map(lambda label: CRF.pair_wise_log_message_passing(2, 1, label, model),
                                  range(len(DECODING_KEY)))
    for i in range(len(DECODING_KEY)):
        print "Log message passing value for m3->2(Y2='%s') = %g" % (DECODING_KEY[i], log_message_passing_val[i])


def solve_q_2_2():
    features_params = parse_model_params(MODEL_PATH_FEATURES)
    transition_params = parse_model_params(MODEL_PATH_TRANSITION)
    print "------------Solution for Q2.2---------------"
    test_file_path = os.path.join(DATA_FOLDER, "test_img1.txt")
    features = parse_features(test_file_path)
    model = linear_chain_model(len(features), features_params, transition_params)
    model.x = features
    model.reset_cache()
    # Updating log_partition
    CRF.update_log_partition(model)
    prob_dist = np.zeros((len(features), len(DECODING_KEY)))
    for position in range(len(features)):
        probability_dist_aray = map(lambda label: CRF.point_marginals(model, position, label), range(len(DECODING_KEY)))
        prob_dist[position] = probability_dist_aray
    print prob_dist
    list_labels = map(lambda x: ENCODING_KEY[x], ['t', 'a', 'h'])
    for position in range(len(features) - 1):
        prob_dist = np.zeros((len(list_labels), len(list_labels)))
        for i in range(len(list_labels)):
            for j in range(len(list_labels)):
                prob_dist[i, j] = CRF.pairwise_marginals(model, position, position + 1, list_labels[i], list_labels[j])
        print "Printing pairwise marginal probabilities for node pair %d,%d" % (position + 1, position + 2)
        print prob_dist


def solve_q_2_3():
    features_params = parse_model_params(MODEL_PATH_FEATURES)
    transition_params = parse_model_params(MODEL_PATH_TRANSITION)
    true_words = open(os.path.join(DATA_FOLDER, "test_words.txt"), 'r').read().splitlines()
    true_predictions = 0
    total_chars = 0
    for i in range(1, 201):
        test_file_path = os.path.join(DATA_FOLDER, "test_img%d.txt" % i)
        actual_word = true_words[i - 1]
        features = parse_features(test_file_path)
        total_chars += len(features)
        model = linear_chain_model(len(features), features_params, transition_params)
        model.x = features
        model.reset_cache()
        CRF.update_log_partition(model)
        prob_dist = np.zeros((len(features), len(DECODING_KEY)))
        for position in range(len(features)):
            probability_dist_aray = map(lambda label: CRF.point_marginals(model, position, label),
                                        range(len(DECODING_KEY)))
            prob_dist[position] = probability_dist_aray
        max_prob_labels = np.argmax(prob_dist, 1)
        predicted_word = map(lambda x: DECODING_KEY[x], max_prob_labels)
        if i < 6:
            print "Predicted word for test image %d word:'%s'is '%s'" % (i, actual_word, "".join(predicted_word))
        actual_labels = np.array(map(lambda x: ENCODING_KEY[x], list(actual_word)))
        true_predictions += sum(actual_labels == max_prob_labels)
    accuracy = 1.0 * true_predictions / total_chars * 100.0
    print "Final Total Accuracy : %0.3f" % (accuracy)


def solve_q_3_5():
    features_params = parse_model_params(MODEL_PATH_FEATURES)
    transition_params = parse_model_params(MODEL_PATH_TRANSITION)
    true_words = open(os.path.join(DATA_FOLDER, "train_words.txt"), 'r').read().splitlines()
    log_likelihood = 0.0
    for file_num in range(1, 51):
        train_file_path = os.path.join(DATA_FOLDER, "train_img%d.txt" % file_num)
        actual_word = true_words[file_num - 1]
        features = parse_features(train_file_path)
        actual_labels = map(lambda x: ENCODING_KEY[x], list(actual_word))
        model = linear_chain_model(len(features), features_params, transition_params)
        model.x = features
        model.reset_cache()
        CRF.update_log_partition(model)
        log_likelihood += CRF.energy(features, actual_labels, model.feature_params,
                                     model.transition_params) - model.log_partition
    avg_log_likelihood = log_likelihood / 50.0
    print "Average log Likelihood for first 50 training images: %g" % avg_log_likelihood


def solve_q4_2():
    x0 = np.random.rand(2)
    res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': False})
    print "Location of Maxima: x = %0.1f and y=%0.1f " % (res.x[0], res.x[1])
    print "Maxium Objective Value : %0.1f" % (rosen(res.x))


if __name__ == '__main__':
    solve_q1_1()
    solve_q1_2()
    solve_q1_3()
    solve_q1_4()
    solve_q1_5()
    solve_q_2_1()
    solve_q_2_2()
    solve_q_2_3()
    solve_q_3_5()
    solve_q4_2()
