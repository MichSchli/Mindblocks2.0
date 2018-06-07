class ComponentSpecifications:

    name = None
    identifier = None
    type = None
    canvas_id = None
    canvas_name = None

    component_type_name = None
    component_type_id = None

    def add(self, key, value):
        if key == "name":
            self.name = value
        elif key == "type":
            self.type = value
        elif key == "canvas_id":
            self.canvas_id = value
        elif key == "canvas_name":
            self.canvas_name = value

    def matches(self, element):
        if self.identifier is not None and self.identifier != element.identifier:
            return False
        elif self.name is not None and self.name != element.name:
            return False
        else:
            return True