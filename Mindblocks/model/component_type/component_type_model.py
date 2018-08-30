from Mindblocks.model.abstract.abstract_model import AbstractModel


class ComponentTypeModel(AbstractModel):

    identifier = None
    name = None
    languages = None

    out_sockets = []
    in_sockets = []

    def __init__(self):
        self.out_sockets = [x for x in self.out_sockets]
        self.in_sockets = [x for x in self.in_sockets]

    def assign_default_value(self, attribute_dict):
        pass

    def get_out_sockets(self):
        return self.out_sockets

    def get_in_sockets(self):
        return self.in_sockets

    def has_batches(self, value, previous_values):
        for inp in previous_values.values():
            if not inp:
                return False
        return True

    def initialize(self, input_dictionary, execution_value, output_value_models, tensorflow_session_model):
        execution_value.initialize_tensorflow_variables(tensorflow_session_model)
        return output_value_models

    def determine_placeholders(self, value, in_socket_names):
        return {k: True for k in in_socket_names}

    def is_used(self, socket_name, value, mode):
        return True