class CreationComponentSpecifications:

    name = None
    identifier = None
    component_type_name = None
    canvas_name = None
    canvas_id = None
    language = None
    graph_id = None

    def add(self, key, value):
        if key == "name":
            self.name = value
        elif key == "type":
            self.component_type_name = value
        elif key == "language":
            self.language = value

    def matches(self, element):
        if self.name is not None and self.name != element.name:
            return False
        if self.identifier is not None and self.identifier != element.identifier:
            return False
        if self.component_type_name is not None and self.component_type_name != element.get_component_type_name():
            return False
        if self.canvas_name is not None and self.canvas_name != element.get_canvas_name():
            return False
        return True
