import numpy as np


class linear_chain_model:
    def __init__(self, length,feature_params,transition_params):
        self.length = length
        self.x = np.zeros(length)
        self.y = np.zeros(length)
        self.feature_params = feature_params
        self.transition_params = transition_params
        self.neighbour_dict = {}
        self.__populate_neighbours()
        self.log_message_passing_cache = {}
        self.log_partition=0.0

    def __populate_neighbours(self):
        for i in range(1, self.length - 1):
            self.neighbour_dict[i] = [i - 1, i + 1]
        self.neighbour_dict[0] = [1]
        self.neighbour_dict[self.length - 1] = [self.length - 2]

    def reset_cache(self):
        self.log_message_passing_cache = {}


