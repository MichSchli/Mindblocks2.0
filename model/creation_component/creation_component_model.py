class CreationComponentModel:

    identifier = None
    name = None

    component_type = None
    component_value = None

    def get_component_type_name(self):
        if self.component_type is None:
            return None
        else:
            return self.component_type.name

    def set_attribute(self, key, value):
        self.component_value[key] = value