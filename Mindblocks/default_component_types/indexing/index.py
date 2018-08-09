from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.index.index_type_model import IndexTypeModel
from Mindblocks.model.value_type.old.index_type import IndexType


class Index(ComponentTypeModel):
    name = "Index"
    out_sockets = ["index"]
    languages = ["python"]

    def initialize_value(self, value_dictionary, language):
        return IndexValue()

    def execute(self, input_dictionary, value, output_value_models, mode):
        output_value_models["index"].assign(value.get_index())
        return output_value_models

    def build_value_type_model(self, input_types, value):
        return {"index": IndexTypeModel()}


class IndexValue(ExecutionComponentValueModel):

    def __init__(self):
        self.index = {"forward": {}, "backward": {}}

    def get_index(self):
        return self.index
