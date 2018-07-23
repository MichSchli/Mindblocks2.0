from model.component_type.component_type_model import ComponentTypeModel


class Index(ComponentTypeModel):
    name = "Index"
    out_sockets = ["index"]
    languages = ["python"]

    def initialize_value(self, value_dictionary):
        return IndexValue()

    def execute(self, input_dictionary, value):
        return {"output": value.get_index()}

    def infer_types(self, input_types, value):
        return {"output": "nonstandard"}

    def infer_dims(self, input_dims, value):
        return {"output": "nonstandard"}


class IndexValue:

    def __init__(self):
        self.index = {"forward": {}, "backward": {}}

    def get_index(self):
        return self.index
