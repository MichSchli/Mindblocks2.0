from Mindblocks.model.value_type.refactored.soft_tensor.soft_tensor_value_model import SoftTensorValueModel


class SoftTensorTypeModel:

    dimensions = None
    soft_by_dimensions = None
    string_type = None

    def __init__(self, dimensions, soft_by_dimensions=None, string_type="float"):
        self.dimensions = dimensions

        if soft_by_dimensions is not None:
            self.soft_by_dimensions = soft_by_dimensions
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

    def set_dimension(self, index, value, is_soft=False):
        self.dimensions[index] = value
        self.soft_by_dimensions[index] = is_soft

    def set_data_type(self, string_type):
        self.string_type = string_type

    def cast(self, string_type):
        cast_copy = self.copy()
        cast_copy.string_type = string_type

        return cast_copy