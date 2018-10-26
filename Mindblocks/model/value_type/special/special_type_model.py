from Mindblocks.model.value_type.index.index_value_model import IndexValueModel
from Mindblocks.model.value_type.special.special_value_model import SpecialValueModel


class SpecialTypeModel:

    name = None

    def initialize_value_model(self, language=None):
        return SpecialValueModel()

    def create_from_tensorflow_output(self, output_tensors):
        value_model = self.initialize_value_model()
        value_model.assign(output_tensors[0])
        return value_model

    def format_tensorflow_value_for_output(self, tensorflow_value):
        return [tensorflow_value.item]

    def get_dimensions(self):
        return None

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name