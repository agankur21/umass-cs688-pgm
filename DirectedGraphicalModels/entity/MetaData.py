import os
import re
import sys

print sys.path


class MetaData:
    def __init__(self, file_path=None):
        self.header_names = []
        self.header_description = []
        self.data_keys_map = {}
        self.field_index_map = {}
        if file_path is not None:
            self.parse_data_info(file_path)

    @staticmethod
    def get_elements(text):
        """
        A static method to find retrieve all the columns
        :param text:
        :return:
        """
        return re.findall(r"(\d+) +([A-Z]+) +([\w ]+) +(.+)", text)

    def parse_data_info(self, file_path='data_info.txt'):
        """
        Parse all the relevant data information form the file automatically
        :param file_path:
        :return:
        """
        file_text = open(file_path).read().splitlines()
        for line in file_text:
            elements = MetaData.get_elements(line)[0]
            self.field_index_map[elements[1]] = int(elements[0]) - 1
            self.header_names.append(elements[1])
            self.header_description.append(elements[2])
            keys = map(lambda x: int(x[1]), re.findall(r"((\d):.+?,?)+", elements[3]))
            self.data_keys_map[elements[1]] = keys


if __name__ == '__main__':
    data = MetaData()
    data.parse_data_info(os.path.join(os.getcwd(), "../../Data/data_info.txt"))
    print data.__dict__
