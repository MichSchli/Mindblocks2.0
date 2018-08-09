from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class LstmCell(ComponentTypeModel):

    name = "LstmCell"
    in_sockets = ["input_x", "previous_c", "previous_h"]
    out_sockets = ["output_c", "output_h"]
    languages = ["tensorflow"]

    def initialize_value(self, value_dictionary):
        return LstmCellValue()


class LstmCellValue(ExecutionComponentValueModel):

    pass