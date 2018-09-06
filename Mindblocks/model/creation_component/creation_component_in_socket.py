from Mindblocks.model.abstract.abstract_creation_model import AbstractCreationModel
from Mindblocks.model.abstract.abstract_model import AbstractModel


class CreationComponentInSocket(AbstractModel, AbstractCreationModel):

    component = None
    name = None
    edge = None

    def __init__(self, component, name):
        AbstractCreationModel.__init__(self)
        self.component = component
        self.name = name

    def get_component(self):
        return self.component

    def set_edge(self, edge):
        self.edge = edge

    def get_name(self):
        return self.name

    def get_description(self):
        return self.component.get_name() + ":" + self.get_name()