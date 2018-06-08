class ComponentModel:

    name = None
    identifier = None
    canvas_id = None
    canvas_name = None
    graph_id = None

    component_type = None

    value = None

    in_sockets = None
    out_sockets = None

    def get_name(self):
        return str(self.name)

    def run_python(self):
        self.component_type.execute(self.value)

    def all_in_edges_satisfied(self):
        count = 0
        for socket in self.in_sockets:
            if socket is not None and socket.is_satisfied():
                count += 1

        return count == 0

    def matched_out_sockets(self):
        return [s for s in self.out_sockets if s is not None]

    def __str__(self):
        return "Component: "+str(self.name)+ " (" + (str(self.component_type.name) if self.component_type is not None else "None") + ") |"+str(self.identifier) +" canvas_name="+str(self.canvas_name)

    def describe(self):
        description = "Component: "+str(self.name)+ " (" + (str(self.component_type.name) if self.component_type is not None else "None") + ") |"+str(self.identifier) + "\n"
        description += "canvas_name=" + str(self.canvas_name) + "\n"
        description += "graph_id=" + str(self.graph_id) + "\n"
        if self.value != None:
            description += self.value.describe() + "\n"

        return description