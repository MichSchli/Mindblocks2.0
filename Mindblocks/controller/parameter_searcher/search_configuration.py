import copy


class SearchConfiguration:

    options = None

    def __init__(self):
        self.options = {}

    def set_options(self, options):
        self.options = options

    def register(self, variable_name, search_index, option):
        if variable_name not in self.options:
            self.options[variable_name] = {}

        self.options[variable_name][search_index] = option

    def copy(self):
        new = SearchConfiguration()
        new.set_options(copy.deepcopy(self.options))
        return new

    def __str__(self):
        return str(self.options)

    def get_affected_variables(self):
        return list(self.options.keys())

    def iterate_options(self):
        for name,option in self.options.items():
            for field, value in option.items():
                yield name, field, value
