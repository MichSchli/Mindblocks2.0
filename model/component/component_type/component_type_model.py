from model.component.component_value_model import ComponentValueModel


class ComponentTypeModel:

    name = None
    identifier = None
    ingoing_sockets = None
    outgoung_sockets = None

    def compile(self, value):
        pass

    def get_name(self):
        return self.name

    def get_new_value(self):
        return ComponentValueModel()