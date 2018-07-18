class ComponentTypeModel:

    identifier = None
    name = None

    out_sockets = None
    in_sockets = None

    def __init__(self):
        self.out_sockets = []
        self.in_sockets = []

    def assign_default_value(self, attribute_dict):
        pass

    def get_out_sockets(self):
        return self.out_sockets

    def get_in_sockets(self):
        return self.in_sockets