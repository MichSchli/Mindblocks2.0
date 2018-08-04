class AbstractValueType:

    value_representation = None

    def cast(self, string_type):
        pass

    def get_tensorflow_placeholder(self):
        pass

    def prepare_for_placeholder(self):
        return self.value_representation