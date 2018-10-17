from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_value_model import SoftTensorValueModel


class SoftTensorTypeModel:

    dimensions = None
    soft_by_dimensions = None
    string_type = None

    def __init__(self, dimensions, soft_by_dimensions=None, string_type="float"):
        self.dimensions = dimensions

        if soft_by_dimensions is not None:
            self.dimensions = soft_by_dimensions
        else:
            self.soft_by_dimensions = [False for _ in self.dimensions]

        self.string_type = string_type

    def initialize_value_model(self):
        tensor_max_lengths = [None for _ in self.soft_by_dimensions]

        return SoftTensorValueModel(self.dimensions,
                                    self.string_type,
                                    tensor_max_lengths)

    def copy(self):
        return SoftTensorTypeModel(self.dimensions,
                                   soft_by_dimensions=self.soft_by_dimensions,
                                   string_type=self.string_type)