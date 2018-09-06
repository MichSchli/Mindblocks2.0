import copy


class AbstractExecutionModel:

    origin = None
    mode = None
    value_model = None

    def set_value_model(self, value_model):
        self.value_model = value_model

    def get_value_model(self):
        return self.value_model

    def set_origin(self, creation_object_model):
        self.origin = creation_object_model

    def set_mode(self, mode):
        self.mode = mode

    def get_value_dictionary(self):
        return copy.deepcopy(self.origin.get_value_dictionary())

    def get_mode(self):
        return self.mode

    def get_origin_identifier(self):
        return self.origin.get_identifier()

    def get_description(self):
        return self.origin.get_description() + "@" + self.mode

    # TODO: This makes execution models default to using updated dictionaries as values
    def initialize_value(self, updated_dict, mode):
        return copy.deepcopy(updated_dict)