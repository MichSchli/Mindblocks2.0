from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel
from Mindblocks.model.value_type.index_type import IndexType


class Index(ComponentTypeModel):
    name = "Index"
    out_sockets = ["index"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return IndexValue()

    def execute(self, input_dictionary, value, mode):
        return {"index": value.get_index()}

    def build_value_type(self, input_types, value):
        return {"index": IndexType()}


class IndexValue(ExecutionComponentValueModel):

    def __init__(self):
        self.index = {"forward": {}, "backward": {}}

    def get_index(self):
        return self.index
