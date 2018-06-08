class CanvasSpecifications:

    name = None
    identifier = None

    def add(self, key, value):
        if key == "name":
            self.name = value
        elif key == "id":
            self.identifier = value

    def matches(self, element):
        if self.identifier is not None and self.identifier != element.identifier:
            return False
        elif self.name is not None and self.name != element.name:
            return False
        else:
            return True