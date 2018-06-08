from model.component.component_value_model import ComponentValueModel


class ComponentTypeModel:

    name = None
    identifier = None

    in_socket_names = []
    out_socket_names = []

    def compile(self, value):
        pass

    def get_name(self):
        return self.name

    def get_new_value(self):
        return ComponentValueModel()

    def in_degree(self):
        return len(self.in_socket_names)

    def out_degree(self):
        return len(self.out_socket_names)

    def get_in_socket_id(self, socket_name):
        for i, name in enumerate(self.in_socket_names):
            if socket_name == name:
                return i

        return None

    def get_out_socket_id(self, socket_name):
        for i, name in enumerate(self.out_socket_names):
            if socket_name == name:
                return i

        return None