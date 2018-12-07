from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import numpy as np
import tensorflow as tf

class VocabMask(ComponentTypeModel):

    name = "VocabMask"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = VocabMaskValue()
        value.language = language
        mask_dimensions = value_dictionary["mask_dimensions"][0][0].split(",")
        mask_dimensions = [int(d) for d in mask_dimensions]
        value.set_mask_dimensions(mask_dimensions)
        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        if value.language == "python":
            print("didnt bother defining ReLu for python")
            exit()
        elif value.language == "tensorflow":
            v = input_dictionary["input"].get_value()

            v = tf.nn.relu(v)
            all_lengths = input_dictionary["input"].get_lengths()

            mask_shape = [1]*(len(all_lengths)-1) + [value.vocab_size]
            mask = np.ones(mask_shape, dtype=np.float32)
            outer_slices = tuple([slice(None,None,1) for _ in range(len(all_lengths)-1)])
            for mask_dim in value.mask_dimensions:
                mask_slice = outer_slices + (mask_dim, )
                mask[mask_slice] = 0

            v = tf.multiply(v, mask)

            replacement = tf.zeros_like(v)

            sth = SoftTensorHelper()
            replaced_v = sth.replace_elements_outside_lengths(v, all_lengths, replacement)

            output_models["output"].assign(replaced_v, length_list=all_lengths)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        value.vocab_size = input_types["input"].get_dimensions()[-1]
        return {"output": input_types["input"].copy()}


class VocabMaskValue(ExecutionComponentValueModel):

    mask_dimensions = None

    def set_mask_dimensions(self, dimensions):
        self.mask_dimensions = dimensions