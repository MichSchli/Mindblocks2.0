from model.creation_component.creation_component_model import CreationComponentModel


class CreationComponentRepository:

    elements = None

    def __init__(self, identifier_repository):
        self.identifier_repository = identifier_repository
        self.elements = {}

    def create(self, specifications):
        identifier = self.identifier_repository.create()
        model = CreationComponentModel()
        model.identifier = identifier
        self.elements[identifier] = model

        model.name = specifications.name

        return model

    def count(self):
        return len(self.elements)

    def get(self, specifications):
        l = []
        for key, element in self.elements.items():
            if specifications.matches(element):
                l.append(element)

        return l