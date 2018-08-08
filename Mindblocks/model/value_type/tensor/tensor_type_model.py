from Mindblocks.model.value_type.sequence_batch.sequence_batch_type_model import SequenceBatchTypeModel
from Mindblocks.model.value_type.tensor.tensor_value_model import TensorValueModel


class TensorTypeModel:

    type = None
    dimensions = None

    cached_placeholder = None

    def __init__(self, type, dimensions):
        self.type = type
        self.dimensions = [d for d in dimensions]
        self.cached_casts = {}

    def initialize_value_model(self):
        return TensorValueModel(self.type, self.dimensions)

    def get_tensorflow_placeholder(self):
        placeholder_model = self.initialize_value_model()

        if self.cached_placeholder is None:
            placeholder_model.initialize_as_tensorflow_placeholder()
            self.cached_placeholder = placeholder_model.value
        else:
            placeholder_model.assign(self.cached_placeholder)

        return placeholder_model

    def get_cached_placeholders(self):
        return [self.cached_placeholder]

    def get_data_type(self):
        return self.type

    def copy(self):
        return TensorTypeModel(self.type, self.dimensions)

    def format_from_tensorflow_output(self, output_tensors):
        value_model = self.initialize_value_model()
        value_model.assign(output_tensors[0])
        return value_model

    def get_dimensions(self):
        return self.dimensions

    def subsample(self, dimension):
        self.dimensions[0] = dimension

    def set_inner_dim(self, dimension):
        self.dimensions[-1] = dimension

    def get_inner_dim(self):
        if len(self.dimensions) == 0:
            return 1
        else:
            return self.dimensions[-1]

    cached_casts = None

    def cast(self, new_type):
        if new_type not in self.cached_casts:
            self.cached_casts[new_type] = TensorTypeModel(new_type, self.dimensions)
        return self.cached_casts[new_type]

    def set_data_type(self, new_type):
        self.type = new_type

    def to_sequence_type(self):
        return SequenceBatchTypeModel(self.type, self.dimensions, None)