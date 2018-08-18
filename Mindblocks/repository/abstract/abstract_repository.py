class AbstractRepository:

    elements = None
    timestamps = None

    def __init__(self, identifier_repository):
        self.identifier_repository = identifier_repository
        self.elements = {}
        self.timestamps = {}

    def __create__(self):
        model = self.__initialize_model__()
        self.add(model)
        
        return model

    def count(self):
        return len(self.elements)

    def add(self, model):
        identifier = self.identifier_repository.create()
        model.identifier = identifier
        self.elements[identifier] = model
        self.timestamps[identifier] = len(self.timestamps)

        return model

    def get(self, specifications, should_sort=True):
        l = []
        for key, element in self.elements.items():
            if specifications.matches(element):
                l.append(element)

        if should_sort:
            order = []
            for key, element in self.elements.items():
                if specifications.matches(element):
                    order.append(self.timestamps[key])

            l = [x for _, x in sorted(zip(order, l))]

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