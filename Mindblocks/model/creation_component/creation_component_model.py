class CreationComponentModel:

    identifier = None
    name = None

    canvas = None
    component_type = None
    component_value = None
    graph = None

    in_sockets = None
    out_sockets = None

    language = None

    def __init__(self):
        self.in_sockets = {}
        self.out_sockets = {}

    def get_graph(self):
        return self.graph

    def get_component_type_name(self):
        if self.component_type is None:
            return None
        else:
            return self.component_type.name

    def place_mark(self, mark, socket):
        socket = self.get_out_socket(socket)
        socket.place_mark(mark)

    def get_marked_sockets(self):
        marked = {}
        for socket in list(self.out_sockets.values()):
            if socket.marked():
                marked[socket.get_mark()] = socket
        return marked

    def get_canvas_name(self):
        if self.canvas is None:
            return None
        else:
            return self.canvas.name

    def set_attribute(self, key, value):
        self.component_value[key] = value

    def get_graph_identifier(self):
        if self.graph is None:
            return None
        else:
            return self.graph.identifier

    def count_out_sockets(self):
        return len(self.out_sockets)

    def count_in_sockets(self):
        return len(self.in_sockets)

    def add_out_socket(self, socket):
        self.out_sockets[socket.name] = socket

    def add_in_socket(self, socket):
        self.in_sockets[socket.name] = socket

    def get_out_socket(self, name):
        print(self.out_sockets)
        print(name)
        if name in self.out_sockets:
            return self.out_sockets[name]
        else:
            return None

    def get_in_socket(self, name):
        if name in self.in_sockets:
            return self.in_sockets[name]
        else:
            return None

    def __str__(self):
        return self.name + " (" +self.component_type.name + ")"