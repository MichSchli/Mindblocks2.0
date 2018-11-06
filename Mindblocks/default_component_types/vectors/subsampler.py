import numpy as np

from Mindblocks.error_handling.types.dimension_mismatch_exception import DimensionMismatchException
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class Subsampler(ComponentTypeModel):

    name = "Subsampler"
    in_sockets = ["indexes", "tensor"]
    out_sockets = ["output"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        return SubsamplerValue()

    def has_batches(self, value, previous_values, mode):
        return previous_values["indexes"]

    def add_along_axis(self, tensor, flat_to_add, axis):
        for i in range(len(tensor.shape)):
            if i < axis:
                flat_to_add = np.expand_dims(flat_to_add, 0)
            elif i == axis:
                pass
            elif i > axis:
                flat_to_add = np.expand_dims(flat_to_add, -1)

        return tensor + flat_to_add

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        tensor_to_index = input_dictionary["tensor"].get_value()
        indexes = input_dictionary["indexes"].get_value()

        range_list = [np.arange(idx_dim) for idx_dim in indexes.shape[:-1]]

        acc = 1
        for j in range(len(range_list), 0, -1):
            acc *= tensor_to_index.shape[j]
            range_list[j - 1] *= acc

        for i, r in enumerate(range_list):
            indexes = self.add_along_axis(indexes, r, i)

        new_shape = list(tensor_to_index.shape)
        new_shape[len(indexes.shape) -1] = indexes.shape[-1]
        leftover_dims = list(tensor_to_index.shape)[len(indexes.shape):]

        flat_indexes = indexes.flatten()
        flat_tensor = tensor_to_index.reshape([-1] + leftover_dims)

        picked_elements = flat_tensor[flat_indexes]
        output_tensor = picked_elements.reshape(new_shape)

        tdx_lengths = input_dictionary["tensor"].get_lengths()
        idx_lengths = input_dictionary["indexes"].get_lengths()

        output_lengths = tdx_lengths[:]
        output_lengths[len(indexes.shape) -1] = idx_lengths[-1]

        for i in range(len(indexes.shape), len(output_lengths)):
            if output_lengths[i] is not None:
                leftover_length_dims = list(output_lengths[i].shape)[len(indexes.shape):]
                new_length_shape = list(output_lengths[i].shape)
                new_length_shape[len(indexes.shape) - 1] = indexes.shape[-1]

                flat_lengths = output_lengths[i].reshape([-1] + leftover_length_dims)
                picked_length_elements = flat_lengths[flat_indexes]
                output_lengths[i] = picked_length_elements.reshape(new_length_shape)

        output_value_models["output"].assign(output_tensor, length_list=output_lengths)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        target_dims = input_types["tensor"].get_dimensions()
        index_dims = input_types["indexes"].get_dimensions()

        dims_to_preserve = index_dims[:-1]

        target_dim_string = input_types["tensor"].get_dimension_string()
        index_dim_string = input_types["indexes"].get_dimension_string()

        target_soft = input_types["tensor"].get_soft_by_dimensions()
        index_soft = input_types["indexes"].get_soft_by_dimensions()

        # Verify preserved dimensions match:
        for i in range(len(dims_to_preserve)):
            if dims_to_preserve[i] is not None and dims_to_preserve[i] != target_dims[i]:
                raise DimensionMismatchException("Mismatched dimensions along axis " + str(i) + " in component "
                                                 + value.get_name() + ": "
                                                 + target_dim_string
                                                 + " cannot be subsampled with index dimensions "
                                                 + index_dim_string
                                                 + ".")

            if index_soft[i] != target_soft[i]:
                raise DimensionMismatchException("Mismatched softness along axis " + str(i) + " in component "
                                                 + value.get_name() + ": "
                                                 + target_dim_string
                                                 + " cannot be subsampled with index dimensions "
                                                 + index_dim_string
                                                 + ".")

        out_dimensions = index_dims + target_dims[len(index_dims):]
        out_data_type = input_types["tensor"].get_data_type()

        out_softness = target_soft[:]
        out_softness[len(index_dims)-1] = index_soft[-1]

        output_type = SoftTensorTypeModel(out_dimensions,
                                          soft_by_dimensions=out_softness,
                                          string_type=out_data_type)
        print(out_dimensions)
        return {"output": output_type}


class SubsamplerValue(ExecutionComponentValueModel):

    pass