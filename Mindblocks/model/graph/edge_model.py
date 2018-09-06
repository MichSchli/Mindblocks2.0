from Mindblocks.model.abstract.abstract_creation_model import AbstractCreationModel
from Mindblocks.model.abstract.abstract_model import AbstractModel


class EdgeModel(AbstractModel, AbstractCreationModel):

    source_socket = None
    target_socket = None
    cast = None
    dropout_rate = None

    def __init__(self, source_socket, target_socket):
        AbstractCreationModel.__init__(self)
        self.source_socket = source_socket
        self.target_socket = target_socket

    def get_source_component_name(self):
        return self.source_socket.get_component().name

    def get_target_component_name(self):
        return self.target_socket.get_component().name

    def get_description(self):
        return self.source_socket.get_description() + " -> " + self.target_socket.get_description()