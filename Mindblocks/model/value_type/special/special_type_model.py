from Mindblocks.model.value_type.index.index_value_model import IndexValueModel
from Mindblocks.model.value_type.special.special_value_model import SpecialValueModel


class SpecialTypeModel:

    def initialize_value_model(self):
        return SpecialValueModel()

    def format_from_tensorflow_output(self, output_tensors):
        value_model = self.initialize_value_model()
        value_model.assign(output_tensors[0])
        return value_model