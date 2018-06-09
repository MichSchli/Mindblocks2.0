from model.component.component_type.component_type_model import ComponentTypeModel


class Add(ComponentTypeModel):

    name = "Add"
    in_socket_names = ["left", "right"]
    out_socket_names = ["output"]
    available_languages = ["python", "tensorflow"]

    def __init__(self):
        pass

    def execute(self, in_sockets, value):
        return [in_sockets[0].get_value() + in_sockets[1].get_value()]