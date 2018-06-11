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

    language = None

    def __init__(self):
        self.out_sockets = []

    def copy(self):
        copy = ComponentModel()

        copy.in_sockets = [None] * len(self.in_sockets)
        copy.out_sockets = [[]] * len(self.out_sockets)

        copy.component_type = self.component_type
        copy.language = self.language
        copy.value = self.value.copy()

        return copy

    def get_name(self):
        return str(self.name)

    def run(self, language="python"):
        component_output = self.component_type.execute([s.get_value() for s in self.in_sockets], self.value)

        outputs = []

        print(component_output)
        print(self.out_sockets)

        for i in range(len(self.out_sockets)):
            for edge in self.out_sockets[i]:
                edge.put_value(component_output[i])
            if len(self.out_sockets[i]) == 0:
                outputs.append(component_output[i])

        return outputs

    def all_in_edges_satisfied(self):
        count = 0
        for socket in self.in_sockets:
            if socket is not None and not socket.is_satisfied():
                count += 1

        return count == 0

    def matched_out_sockets(self):
        return [s for s in self.out_sockets if s is not []]

    def __str__(self):
        return "Component: "+str(self.get_name())+ " (" + (str(self.component_type.name) if self.component_type is not None else "None") + ") |"+str(self.identifier) +" canvas_name="+str(self.canvas_name)

    def describe(self):
        description = "Component: "+str(self.name)+ " (" + (str(self.component_type.name) if self.component_type is not None else "None") + ") |"+str(self.identifier) + "\n"
        description += "canvas_name=" + str(self.canvas_name) + "\n"
        description += "graph_id=" + str(self.graph_id) + "\n"
        if self.value != None:
            description += self.value.describe() + "\n"

        return description