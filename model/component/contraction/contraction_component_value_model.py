from model.component.component_value_model import ComponentValueModel


class ContractionComponentValueModel(ComponentValueModel):

    source_component = None
    source_sockets = None
    target_component = None
    target_sockets = None

    source_outputs = None

    def __init__(self):
        self.source_outputs = []

    def copy(self):
        copy = ContractionComponentValueModel()
        copy.source_sockets = self.source_sockets
        copy.target_sockets = self.target_sockets
        copy.source_component = self.source_component.copy()
        copy.target_component = self.target_component.copy()
        copy.source_outputs = self.source_outputs
        return copy

    def add_source_output(self, socket):
        if not socket in self.source_outputs:
            self.source_outputs.append(socket)

    def distribute_to_target(self, c1_output, in_sockets):
        input_pointer = 0

        distributed_input = [None] * len(self.target_component.in_sockets)

        for i in range(len(self.target_component.in_sockets)):
            if i not in self.target_sockets:
                distributed_input[i] = in_sockets[input_pointer]
                input_pointer += 1
            else:
                for source_socket, target_socket in zip(self.source_sockets, self.target_sockets):
                    if target_socket == i:
                        distributed_input[i] = c1_output[source_socket]

        return distributed_input

    def distribute_to_output(self, c1_output):
        true_output = []
        for i in self.source_outputs:
            true_output.append(c1_output[i])

        return true_output
