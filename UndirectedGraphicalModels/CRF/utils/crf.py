import numpy as np

from config.config import *


class CRF():
    @staticmethod
    def energy(x, y, feature_params, transition_params):
        # Compute energy defined in question 1.2
        # inputs: input x, label sequence y, params
        # output: energy (a number)
        if len(x) != len(y):
            raise Exception("Mismatch in dimension between x and y")
        total_node_log_potential = CRF.log_sequence_potential(x, feature_params, y)
        total_pairwise_log_potential = 0.0
        for i in range(len(y) - 1):
            total_pairwise_log_potential += CRF.log_pairwise_potential(y[i], y[i + 1], transition_params)
        return total_pairwise_log_potential + total_node_log_potential

    @staticmethod
    def log_sequence_potential(x, feature_params, y=None):
        F = feature_params.shape[1]
        if len(x[0]) != F:
            raise TypeError("Incorrect dimensions of the feature")
        log_potential_matrix = x.dot(feature_params.T)
        if y is None:
            return log_potential_matrix
        else:
            log_potential = 0.0
            for i in range(len(y)):
                log_potential += 1.0 * log_potential_matrix[i][int(y[i])]
            return log_potential

    @staticmethod
    def log_pairwise_potential(y1, y2, transition_params):
        if y1 >= len(transition_params) or y2 >= len(transition_params):
            raise IndexError("Incorrect Indices")
        return transition_params[int(y1)][int(y2)]

    @staticmethod
    def log_node_potential(label, index, feature_params, x):
        return x[index].dot(feature_params[label])

    @staticmethod
    def log_partition_naive(x, all_y, feature_params, transition_params):
        partition = 0.0
        for y in all_y:
            partition += np.exp(CRF.energy(x, y, feature_params, transition_params))
        return np.log(partition)

    @staticmethod
    def pair_wise_log_message_passing(t, s, label_s, model):
        if (t, s, label_s) not in model.log_message_passing_cache:
            message_passing_val = 0.0
            for label_t in range(len(DECODING_KEY)):
                log_psi_t = CRF.log_node_potential(label_t, t, model.feature_params, model.x)
                log_psi_s_t = CRF.log_pairwise_potential(label_s, label_t, model.transition_params)
                log_neighbouring_message_passing = log_psi_t + log_psi_s_t
                for u in model.neighbour_dict[t]:
                    if u != s:
                        log_neighbouring_message_passing += CRF.pair_wise_log_message_passing(u, t, label_t, model)
                message_passing_val += np.exp(log_neighbouring_message_passing)
            model.log_message_passing_cache[(t, s, label_s)] = np.log(message_passing_val)
        return model.log_message_passing_cache[(t, s, label_s)]

    @staticmethod
    def update_log_partition(model):
        partition = 0.0
        for label in range(len(DECODING_KEY)):
            log_node_potential = CRF.log_node_potential(label, 0, model.feature_params, model.x)
            for t in model.neighbour_dict[0]:
                if t == 0:
                    continue
                log_node_potential += CRF.pair_wise_log_message_passing(t, 0, label, model)
            partition += np.exp(log_node_potential)
        model.log_partition = np.log(partition)

    @staticmethod
    def point_marginals(model, s, label):
        if model.log_partition == 0.0:
            CRF.update_log_partition(model)
        log_psi = CRF.log_node_potential(label, s, model.feature_params, model.x)
        log_message = 0.0
        for t in model.neighbour_dict[s]:
            log_message += CRF.pair_wise_log_message_passing(t, s, label, model)
        return np.exp(log_psi + log_message - model.log_partition)

    @staticmethod
    def pairwise_marginals(model, s, t, label_s, label_t):
        if s not in model.neighbour_dict[t] or t not in model.neighbour_dict[s]:
            raise Exception("Not neighbours or something wrong in the neighbour dictionary")
        else:
            log_psi = CRF.log_node_potential(label_s, s, model.feature_params, model.x) + CRF.log_node_potential(
                label_t, t, model.feature_params, model.x) + CRF.log_pairwise_potential(label_s, label_t,
                                                                                        model.transition_params)
            log_message = 0.0
            for u in model.neighbour_dict[s]:
                if u ==t :
                    continue
                log_message += CRF.pair_wise_log_message_passing(u, s, label_s, model)
            for u in model.neighbour_dict[t]:
                if u ==s :
                    continue
                log_message += CRF.pair_wise_log_message_passing(u, t, label_t, model)
        return np.exp(log_psi + log_message - model.log_partition)


    @staticmethod
    def predict(x, params):
        pass
