from model.component.component_model import ComponentModel
from model.component.component_value_model import ComponentValueModel
from model.component.contraction.contraction_component_type_model import ContractionComponentTypeModel
from model.component.contraction.contraction_component_value_model import ContractionComponentValueModel


class ContractionComponentModel(ComponentModel):

    def __init__(self, edge):
        self.value = ContractionComponentValueModel()
        self.value.source_component = edge.source
        self.value.source_sockets = [edge.source_socket]
        self.value.target_component = edge.target
        self.value.target_sockets = [edge.target_socket]

        self.component_type = ContractionComponentTypeModel()

    def contract_additional_edge(self, edge):
        self.value.source_sockets.append(edge.source_socket)
        self.value.target_sockets.append(edge.target_socket)

    def add_source_output(self, socket):
        self.value.add_source_output(socket)

    def get_name(self):
        return "meta(" + self.value.source_component.get_name() + "," + self.value.target_component.get_name() + ")"