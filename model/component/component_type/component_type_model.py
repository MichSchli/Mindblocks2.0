class ComponentTypeModel:

    name = None
    identifier = None
    ingoing_sockets = None
    outgoung_sockets = None

    def compile(self, value):
        pass

    def get_name(self):
        return self.name