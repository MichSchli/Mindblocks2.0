class CreationComponentModel:

    identifier = None
    name = None

    component_type = None

    def get_component_type_name(self):
        if self.component_type is None:
            return None
        else:
            return self.component_type.name