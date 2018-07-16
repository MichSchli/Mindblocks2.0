class CreationComponentSpecifications:

    name = None
    identifier = None
    component_type_name = None

    def add(self, key, value):
        if key == "name":
            self.name = value

    def matches(self, element):
        if self.name is not None and self.name != element.name:
            return False
        if self.identifier is not None and self.identifier != element.identifier:
            return False
        if self.component_type_name is not None and self.component_type_name != element.get_component_type_name():
            return False
        return True
