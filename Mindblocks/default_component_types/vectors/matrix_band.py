from Mindblocks.helpers.soft_tensors.soft_tensor_helper import SoftTensorHelper
from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
import tensorflow as tf

class MatrixBand(ComponentTypeModel):

    name = "MatrixBand"
    in_sockets = ["input"]
    out_sockets = ["output"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = MatrixBandValue()
        value.language = language

        if "row_band_end" in value_dictionary:
            row_band_end = int(value_dictionary["row_band_end"][0][0])
            value.set_row_band_end(row_band_end)

        if "column_band_end" in value_dictionary:
            column_band_end = int(value_dictionary["column_band_end"][0][0])
            value.set_column_band_end(column_band_end)

        return value

    def execute(self, execution_component, input_dictionary, value, output_models, mode):
        v = input_dictionary["input"].get_value()
        lengths = input_dictionary["input"].get_lengths()[:]

        v = tf.matrix_band_part(v, value.row_band_end, value.column_band_end, name=value.get_name())

        output_models["output"].assign(v, length_list=lengths)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        out_type = input_types["input"].copy()

        return {"output": out_type}


class MatrixBandValue(ExecutionComponentValueModel):

    def __init__(self):
        self.row_band_end = -1
        self.column_band_end = -1

    def set_row_band_end(self, index):
        self.row_band_end = index

    def set_column_band_end(self, index):
        self.column_band_end = index