from Mindblocks.model.value_type.soft_tensor.soft_tensor_tf_output_manager import SoftTensorTfOutputManager
from Mindblocks.model.value_type.soft_tensor.soft_tensor_value_model import SoftTensorValueModel

from Mindblocks.model.value_type.soft_tensor.soft_tensor_tf_input_manager import SoftTensorTfInputManager


class SoftTensorTypeModel:

    dimensions = None
    soft_by_dimensions = None
    string_type = None

    placeholder_manager = None
    tf_output_manager = None

    name = None

    reference_name = None

    def __init__(self, dimensions, soft_by_dimensions=None, string_type="float", reference_name="soft_tensor"):
        self.dimensions = dimensions[:]

        if soft_by_dimensions is not None:
            self.soft_by_dimensions = soft_by_dimensions[:]
        else:
            self.soft_by_dimensions = [False for _ in self.dimensions]

        self.set_data_type(string_type)
        self.cached_casts = {}
        self.reference_name = reference_name

    def initialize_value_model(self, language):
        tensor_dimensions = self.get_dimensions()[:]
        tensor_max_lengths = self.get_dimensions()[:]

        return SoftTensorValueModel(tensor_dimensions,
                                    self.string_type,
                                    tensor_max_lengths,
                                    self.soft_by_dimensions,
                                    language)

    def get_tensorflow_placeholder(self):
        if self.placeholder_manager is None:
            self.placeholder_manager = SoftTensorTfInputManager(self.dimensions, self.soft_by_dimensions, self.string_type, self.reference_name)

        placeholder_value = self.initialize_value_model(language="tensorflow")
        self.placeholder_manager.assign_placeholders(placeholder_value)

        return placeholder_value

    def get_cached_placeholders(self):
        if self.placeholder_manager is None:
            self.placeholder_manager = SoftTensorTfInputManager(self.dimensions, self.soft_by_dimensions, self.string_type, self.reference_name)

        return self.placeholder_manager.get_placeholders()

    def format_for_tensorflow_input(self, value):
        if self.placeholder_manager is None:
            self.placeholder_manager = SoftTensorTfInputManager(self.dimensions, self.soft_by_dimensions, self.string_type, self.reference_name)

        return self.placeholder_manager.format_for_input(value)

    def create_from_tensorflow_output(self, tensorflow_output):
        if self.tf_output_manager is None:
            self.tf_output_manager = SoftTensorTfOutputManager(self.dimensions, self.soft_by_dimensions, self.string_type)

        value = self.initialize_value_model(language="python")
        self.tf_output_manager.assign_tensorflow_output(value, tensorflow_output)
        return value

    def format_tensorflow_value_for_output(self, tensorflow_value):
        if self.tf_output_manager is None:
            self.tf_output_manager = SoftTensorTfOutputManager(self.dimensions, self.soft_by_dimensions, self.string_type)

        return self.tf_output_manager.format_for_output(tensorflow_value)

    def copy(self, reference_name=None):
        if reference_name is None:
            reference_name = self.reference_name + "_copy"
        return SoftTensorTypeModel(self.dimensions,
                                   soft_by_dimensions=self.soft_by_dimensions,
                                   string_type=self.string_type,
                                   reference_name=reference_name)

    def get_subtype(self, keep_dims):
        new_dimensions = [self.dimensions[x] for x in keep_dims]
        new_soft = [self.soft_by_dimensions[x] for x in keep_dims]

        return SoftTensorTypeModel(new_dimensions,
                                   soft_by_dimensions=new_soft,
                                   string_type=self.string_type)

    def set_dimension(self, index, value, is_soft=None):
        if self.is_scalar():
            self.dimensions = [None]
            self.soft_by_dimensions = [None]

        self.dimensions[index] = value
        if is_soft is not None:
            self.soft_by_dimensions[index] = is_soft

    def get_dimension(self, index):
        return self.dimensions[index]

    def get_dimensions(self):
        return self.dimensions

    def dimension_to_str(self, idx):
        dim = self.get_dimensions()[idx]
        is_soft = self.get_soft_by_dimensions()[idx]

        soft_suffix =  "(s)" if is_soft else ""

        if dim is not None:
            return str(dim) + soft_suffix
        else:
            return "U" + soft_suffix

    def get_dimension_string(self):
        dimensions_to_str = [self.dimension_to_str(i) for i in range(len(self.get_dimensions()))]
        return ",".join(dimensions_to_str)

    def get_soft_by_dimensions(self):
        return self.soft_by_dimensions

    def add_dimension(self, index, dimension, is_soft=False):
        if index >= 0:
            self.dimensions.insert(index, dimension)
            self.soft_by_dimensions.insert(index, is_soft)
        elif index == -1:
            self.dimensions.append(dimension)
            self.soft_by_dimensions.append(is_soft)
        elif index < -1:
            self.dimensions.insert(index + 1, dimension)
            self.soft_by_dimensions.insert(index + 1, is_soft)

    def delete_dimension(self, index):
        del self.dimensions[index]
        del self.soft_by_dimensions[index]

    def is_scalar(self):
        return len(self.dimensions) == 0

    def set_data_type(self, string_type):
        self.string_type = string_type

    def get_data_type(self):
        return self.string_type

    cached_casts = None

    def cast(self, string_type):
        if string_type not in self.cached_casts:
            cast_copy = self.copy()
            cast_copy.set_data_type(string_type)
            self.cached_casts[string_type] = cast_copy

        return self.cached_casts[string_type]

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name