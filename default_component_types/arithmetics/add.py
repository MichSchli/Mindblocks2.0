from model.component_type.component_type_model import ComponentTypeModel


class Add(ComponentTypeModel):

    name = "Add"
    in_sockets = ["left", "right"]
    out_sockets = ["output"]
    languages = ["tensorflow", "python"]

    def initialize_value(self, value_dictionary):
        return AddValue()

    def execute(self, input_dictionary, value):
        return {"output": input_dictionary["left"] + input_dictionary["right"]}

class AddValue:

    pass