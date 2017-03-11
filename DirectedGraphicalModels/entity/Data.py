import pandas as pd


class Data:
    @staticmethod
    def get_all_combinations(list_values, list_index, temp_list, out):
        """
        A generic function which calculates for all possible combinations of the n-tuple from the set of possible values
        e.g. [[1,2],[3,4],[10,11]] yields [[1,3,10],[1,3,11],[1,4,10],[1,4,11],[2,3,10],[2,3,11],[2,4,10],[2,4,11]]
        The key point is maintain the index consistency
        :param list_values: A list of list containing all possible values for each column
        :param list_index: starting list index
        :param temp_list: a temp list to store a particular combination
        :param out: output
        :return: None
        """
        if list_index >= len(list_values):
            out.append(list(temp_list))
            return
        else:
            for y in range(len(list_values[list_index])):
                temp_list.append(list_values[list_index][y])
                Data.get_all_combinations(list_values, list_index + 1, temp_list, out)
                temp_list.pop()

    def __init__(self, meta_data=None):
        self.meta_data = meta_data
        self.data = pd.DataFrame(columns=meta_data.header_names)

    def get_filter_map_list(self, columns):
        """
        Get a the possible values
        :param columns:
        :return:
        """
        list_values = map(lambda column: self.meta_data.data_keys_map[column], columns)
        return list_values

    def populate_data(self, file_path):
        """
        Populate the data field from he csv file
        :param file_path: Path of the csv data file
        :return: None
        """
        df = pd.read_csv(file_path, names=self.meta_data.header_names)
        self.data = self.data.append(df)

    def filter(self, filter_keys, filter_values):
        """
        The function filters the data on the basis of multiple filter conditions
        :param filter_keys: The filter keys representing the columns to filter
        :param filter_values: The filter value for each column
        :return:
        """
        boolean_df = None
        for i in range(len(filter_keys)):
            key = filter_keys[i]
            value = filter_values[i]
            if boolean_df is not None:
                boolean_df = boolean_df & (self.data[key] == value)
            else:
                boolean_df = self.data[key] == value
        return self.data[boolean_df]

    def select(self, filtered_df, select_column):
        """
        A probability distribution for the column to be selected from the filtered data
        :param filtered_df:
        :param select_column:
        :return:
        """
        column = filtered_df[select_column]
        count_map = dict(column.value_counts())
        total_count = column.count()
        prob_map = {int(key): count_map[key] * 1.0 / total_count for key in count_map}
        return prob_map


    def get_probability_map(self, select_column, filter_columns=None):
        """
        The function calculates for the probability distribution for a child given its multiple parents
        :param select_column: The child node
        :param filter_columns:
        :return:
        """
        prob_map = {}
        if filter_columns is not None:
            filter_combination_values = []
            temp_list = []
            Data.get_all_combinations(self.get_filter_map_list(filter_columns), 0, temp_list, filter_combination_values)
            for filter_values in filter_combination_values:
                filtered_df = self.filter(filter_columns, filter_values)
                cpd = self.select(filtered_df, select_column)
                if len(filter_values) > 1:
                    prob_map[tuple(filter_values)] = cpd
                else:
                    prob_map[filter_values[0]] = cpd
        else:
            return self.select(self.data, select_column)
        return prob_map
