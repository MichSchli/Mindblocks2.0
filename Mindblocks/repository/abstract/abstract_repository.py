class AbstractRepository:

    elements = None

    def __init__(self, identifier_repository):
        self.identifier_repository = identifier_repository
        self.elements = {}

    def __create__(self):
        identifier = self.identifier_repository.create()
        model = self.__initialize_model__()
        model.identifier = identifier
        self.elements[identifier] = model

        return model

    def count(self):
        return len(self.elements)

    def add(self, model):
        identifier = self.identifier_repository.create()
        model.identifier = identifier
        self.elements[identifier] = model

        return model

    def get(self, specifications):
        l = []
        for key, element in self.elements.items():
            if specifications.matches(element):
                l.append(element)

        return l

    def get_by_name(self, name):
        spec = self.get_specifications()
        spec.name = name
        return self.get(spec)

    def get_all(self):
        return list(self.elements.values())

    def delete(self, element):
        identifier = element.identifier

        if identifier is not None:
            del self.elements[identifier]

    def print_elements(self):
        print("Printing elements:")
        print("==================")

        for k,v in self.elements.items():
            print(str(k) + ": " + str(v))