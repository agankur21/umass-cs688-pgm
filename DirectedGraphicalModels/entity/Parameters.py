class Parameters:
    """
    A Generic class to store parameters for each graphical model
    """

    def __init__(self, data, parent_info=None):
        self.data = data
        self.probability_g = None
        self.probability_a = None
        self.probability_bp = None
        self.probability_ch = None
        self.probability_hd = None
        self.probability_cp = None
        self.probability_eia = None
        self.probability_ecg = None
        self.probability_hr = None
        self.param_list = ['A', 'G', 'BP', 'CH', 'HD', 'CP', 'EIA', 'ECG', 'HR']
        self.parent_info = parent_info
        self.__populate_parameters()

    def get_probability_distribution(self, column_name):
        if column_name not in self.parent_info or len(self.parent_info[column_name]) == 0:
            return self.data.get_probability_map(column_name)
        else:
            return self.data.get_probability_map(column_name, self.parent_info[column_name])

    def __populate_parameters(self):
        if self.parent_info is None:
            self.probability_a = self.data.get_probability_map('A')
            self.probability_g = self.data.get_probability_map('G')
            self.probability_bp = self.data.get_probability_map('BP', ['G'])
            self.probability_ch = self.data.get_probability_map('CH', ['G', 'A'])
            self.probability_hd = self.data.get_probability_map('HD', ['BP', 'CH'])
            self.probability_cp = self.data.get_probability_map('CP', ['HD'])
            self.probability_eia = self.data.get_probability_map('EIA', ['HD'])
            self.probability_ecg = self.data.get_probability_map('ECG', ['HD'])
            self.probability_hr = self.data.get_probability_map('HR', ['HD', 'A'])
        else:
            self.probability_a = self.get_probability_distribution('A')
            self.probability_g = self.get_probability_distribution('G')
            self.probability_bp = self.get_probability_distribution('BP')
            self.probability_ch = self.get_probability_distribution('CH')
            self.probability_hd = self.get_probability_distribution('HD')
            self.probability_cp = self.get_probability_distribution('CP')
            self.probability_eia = self.get_probability_distribution('EIA')
            self.probability_ecg = self.get_probability_distribution('ECG')
            self.probability_hr = self.get_probability_distribution('HR')

    def __extract_probability(self, column_name, probability_distribution, value_dict):
        if column_name not in self.parent_info or len(self.parent_info[column_name]) == 0:
            return probability_distribution[value_dict[column_name]]
        elif len(self.parent_info[column_name]) == 1:
            return probability_distribution[value_dict[self.parent_info[column_name][0]]][value_dict[column_name]]
        else:
            value_tuple = tuple(map(lambda x: value_dict[x], self.parent_info[column_name]))
            return probability_distribution[value_tuple][value_dict[column_name]]

    def get_point_probability(self, value_dict):
        if self.parent_info is None:
            output = 1.0 * self.probability_a[value_dict['A']] * self.probability_g[value_dict['G']] * \
                     self.probability_bp[value_dict['G']][value_dict['BP']] * \
                     self.probability_ch[(value_dict['G'], value_dict['A'])][value_dict['CH']] * \
                     self.probability_hd[(value_dict['BP'], value_dict['CH'])][value_dict['HD']] * \
                     self.probability_cp[value_dict['HD']][value_dict['CP']] * \
                     self.probability_eia[value_dict['HD']][value_dict['EIA']] * \
                     self.probability_ecg[value_dict['HD']][value_dict['ECG']] * \
                     self.probability_hr[(value_dict['HD'], value_dict['A'])][value_dict['HR']]
        else:
            output = 1.0 * self.__extract_probability('A', self.probability_a, value_dict) * self.__extract_probability(
                'G', self.probability_g, value_dict) * self.__extract_probability(
                'BP', self.probability_bp, value_dict) * self.__extract_probability(
                'CH', self.probability_ch, value_dict) * self.__extract_probability(
                'HD', self.probability_hd, value_dict) * self.__extract_probability(
                'CP', self.probability_cp, value_dict) * self.__extract_probability(
                'EIA', self.probability_eia, value_dict) * self.__extract_probability(
                'ECG', self.probability_ecg, value_dict) * self.__extract_probability(
                'HR', self.probability_hr, value_dict)
        return output

    def predict(self, value_dict_conditionals, predict_column):
        values_columns = self.data.meta_data.data_keys_map[predict_column]
        max_class = 0
        max_probability = 0
        for value in values_columns:
            value_dict_conditionals[predict_column] = value
            point_probability = self.get_point_probability(value_dict_conditionals)
            if point_probability > max_probability:
                max_class = value
                max_probability = point_probability
        return max_class
