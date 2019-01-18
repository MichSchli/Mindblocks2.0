import numpy as np
import tensorflow as tf

from Mindblocks.error_handling.types.dimension_mismatch_exception import DimensionMismatchException
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.soft_tensor.soft_tensor_type_model import SoftTensorTypeModel


class ScatterLike(ComponentTypeModel):

    name = "ScatterLike"
    in_sockets = ["indexes", "shape"]
    out_sockets = ["output"]
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        return ScatterLikeValue(language, value_dictionary["constant_value"][0][0])

    def execute(self, execution_component, input_dictionary, value, output_value_models, mode):
        indexes = input_dictionary["indexes"].get_value()
        shape_tensor = tf.shape(input_dictionary["shape"].get_value())
        lengths = input_dictionary["shape"].get_lengths()

        values = tf.ones_like(indexes[:,0], dtype=tf.float32) * value.static_value
        scattered = tf.scatter_nd(indexes, values, shape_tensor)

        output_value_models["output"].assign(scattered, length_list=lengths)
        return output_value_models

    def build_value_type_model(self, input_types, value, mode):
        output_type = input_types["shape"].copy()

        return {"output": output_type}


class ScatterLikeValue(ExecutionComponentValueModel):

    static_value = None

    def __init__(self, language, static_value):
        self.language = language
        self.static_value = float(static_value)