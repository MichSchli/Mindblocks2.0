from Mindblocks.model.abstract.abstract_creation_model import AbstractCreationModel
from Mindblocks.model.abstract.abstract_model import AbstractModel


class CreationComponentOutSocket(AbstractModel, AbstractCreationModel):

    component = None
    name = None
    edges = None
    mark = None

    def __init__(self, component, name):
        AbstractCreationModel.__init__(self)
        self.component = component
        self.name = name
        self.edges = []

    def get_component(self):
        return self.component

    def add_edge(self, edge):
        self.edges.append(edge)

    def place_mark(self, mark):
        self.mark = mark

    def marked(self):
        return self.mark is not None

    def get_mark(self):
        return self.mark

    def get_name(self):
        return self.name

    def get_description(self):
        return self.component.get_name() + ":" + self.get_name()