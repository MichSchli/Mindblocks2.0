class AbstractCreationModel:

    value_dictionary = None

    def __init__(self):
        self.value_dictionary = {}

    def add_value_line(self, key, item):
        if key not in self.value_dictionary:
            self.value_dictionary[key] = [item]
        else:
            self.value_dictionary[key].append(item)

    def add_value_lines(self, key, item):
        if key not in self.value_dictionary:
            self.value_dictionary[key] = item[:]
        else:
            self.value_dictionary[key].extend(item)

    def get_value_dictionary(self):
        return self.value_dictionary