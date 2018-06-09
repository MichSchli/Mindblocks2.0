class ComponentTypeSpecifications:

    name = None
    identifier = None
    available_languages = None

    def matches(self, element):
        if self.identifier is not None and self.identifier != element.identifier:
            return False
        elif self.name is not None and self.name != element.name:
            return False
        else:
            return True