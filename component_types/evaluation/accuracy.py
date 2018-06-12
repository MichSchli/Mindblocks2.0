from model.component.component_type.component_type_model import ComponentTypeModel
from model.graph.value_type_model import ValueTypeModel
import numpy as np

class Accuracy(ComponentTypeModel):

    name = "Accuracy"
    in_socket_names = ["predictions", "labels"]
    out_socket_names = ["output"]
    available_languages = ["python"]

    def __init__(self):
        pass

    def execute(self, in_sockets, value, language="python"):
        preds = np.squeeze(in_sockets[0])
        labels = np.squeeze(in_sockets[1])

        equals = np.sum(preds == labels)
        accuracy = equals / np.shape(labels)[0]

        return [accuracy]