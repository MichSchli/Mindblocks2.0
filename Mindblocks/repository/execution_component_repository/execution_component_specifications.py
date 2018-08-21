class ExecutionComponentSpecifications:

    identifier = None
    name = None
    mode = None

    def matches(self, element):
        if self.identifier is not None and self.identifier != element.identifier:
            return False
        if self.name is not None and self.name != element.name:
            return False
        if self.mode is not None and self.mode != element.mode:
            return False
        return True
