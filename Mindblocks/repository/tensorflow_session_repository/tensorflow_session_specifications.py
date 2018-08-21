class TensorflowSessionSpecifications:

    identifier = None

    def matches(self, element):
        if self.identifier is not None and self.identifier != element.identifier:
            return False
        return True
