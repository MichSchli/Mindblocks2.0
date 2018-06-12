from model.component.component_type.component_type_model import ComponentTypeModel
import numpy as np

class Argmax(ComponentTypeModel):

    name = "ArgMax"
    in_socket_names = ["input"]
    out_socket_names = ["output"]
    available_languages = ["python"]

    def __init__(self):
        pass

    def execute(self, in_sockets, value, language="python"):
        return [np.argmax(in_sockets[0], axis=1)]