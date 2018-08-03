class GraphSpecifications:
    name = None
    identifier = None

    def add(self, key, value):
        if key == "name":
            self.name = value

    def add_all(self, dictionary):
        for key, value in dictionary.items():
            self.add(key, value)

    def matches(self, element):
        if self.name is not None and self.name != element.name:
            return False
        if self.identifier is not None and self.identifier != element.identifier:
            return False
        return True

