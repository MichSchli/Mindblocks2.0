from model.component.component_type.component_type_model import ComponentTypeModel
from model.component.component_value_model import ComponentValueModel
from model.graph.value_type_model import ValueTypeModel


class DataSplitter(ComponentTypeModel):

    name = "DataSplitter"
    in_socket_names = ["input"]
    out_socket_names = ["left_output", "right_output"]

    def get_new_value(self):
        return DataSplitterValue()

    def execute(self, in_sockets, value, language="python"):
        return [value.get_left_outputs(in_sockets[0]), value.get_right_outputs(in_sockets[0])]

    def evaluate_value_type(self, in_types, value):
        data_type = in_types[0].type
        data_shape_left = [None]*len(in_types[0].dim)
        data_shape_right = [None]*len(in_types[0].dim)

        for i in range(len(in_types[0].dim)):
            data_shape_left[i] = in_types[0].dim[i]
            data_shape_right[i] = in_types[0].dim[i]

        data_shape_left[-1] = len(value.left_columns)
        data_shape_right[-1] = len(value.right_columns)

        return [ValueTypeModel(data_type, data_shape_left), ValueTypeModel(data_type, data_shape_right)]

class DataSplitterValue(ComponentValueModel):

    left_columns = None
    right_columns = None

    def load(self, value_lines):
        if "left_columns" in value_lines:
            self.left_columns = [int(i) for i in value_lines["left_columns"][0][0].split(",")]
        if "right_columns" in value_lines:
            self.right_columns = [int(i) for i in value_lines["right_columns"][0][0].split(",")]

    def get_left_outputs(self, input_tensor):
        return [[row[i] for i in self.left_columns] for row in input_tensor]

    def get_right_outputs(self, input_tensor):
        return [[row[i] for i in self.right_columns] for row in input_tensor]

    def describe(self):
        return "value=\""+self.value+"\""

    def copy(self):
        copy = DataSplitterValue()
        copy.left_columns = self.left_columns
        copy.right_columns = self.right_columns
        return copy
