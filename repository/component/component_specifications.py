class ComponentSpecifications:

    name = None
    identifier = None
    canvas_id = None
    canvas_name = None

    component_type_name = None
    component_type_id = None

    graph_id = None
    language = None

    def add(self, key, value):
        if key == "name":
            self.name = value
        elif key == "canvas_id":
            self.canvas_id = value
        elif key == "canvas_name":
            self.canvas_name = value
        elif key == "type":
            self.component_type_name = value
        elif key == "type_id":
            self.component_type_id = value
        elif key == "language":
            self.language = value

    def matches(self, element):
        if self.identifier is not None and self.identifier != element.identifier:
            return False
        elif self.name is not None and self.name != element.name:
            return False
        else:
            return True