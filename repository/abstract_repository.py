class AbstractRepository:

    identifier_repository = None

    elements = None
    id_map = None

    def __init__(self, identifier_repository):
        self.identifier_repository = identifier_repository
        self.elements = []
        self.id_map = {}

    def get(self, specifications):
        if specifications.identifier is not None:
            if specifications.identifier in self.id_map:
                return [self.id_map[specifications.identifier]]
            else:
                return []
        else:
            return_vals = []
            for element in self.elements:
                if specifications.matches(element):
                    return_vals.append(element)

            return return_vals

    def __create__(self, element):
        self.elements.append(element)
        self.id_map[element.identifier] = element