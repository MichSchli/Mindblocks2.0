class ComponentTypeModel:

    identifier = None
    name = None
    languages = None

    out_sockets = []
    in_sockets = []

    def __init__(self):
        self.out_sockets = [x for x in self.out_sockets]
        self.in_sockets = [x for x in self.in_sockets]

    def assign_default_value(self, attribute_dict):
        pass

    def get_out_sockets(self):
        return self.out_sockets

    def get_in_sockets(self):
        return self.in_sockets