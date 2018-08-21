from Mindblocks.repository.graph.graph_specifications import GraphSpecifications


class ValueDictionaryBuilder:

    variable_repository = None

    def __init__(self, variable_repository, graph_repository):
        self.variable_repository = variable_repository
        self.graph_repository = graph_repository

    def get_all_variables(self):
        return self.variable_repository.get_all()

    def build_value_dictionary(self, runs, run_modes, existing_dictionary=None):
        if not existing_dictionary:
            value_dictionary = {}
        else:
            value_dictionary = existing_dictionary

        activated_output_sockets = []
        for run, run_mode in zip(runs, run_modes):
            for socket in run:
                activated_output_sockets.append((socket, run_mode))

        while len(activated_output_sockets) > 0:
            socket, run_mode = activated_output_sockets.pop()
            component = socket.get_component()

            if component.identifier not in value_dictionary:
                value_dictionary[component.identifier] = {}

            unique_modes = self.list_unique_modes(component)
            if run_mode in unique_modes:
                if run_mode not in value_dictionary[component.identifier]:
                    initialized_value = self.initialize_value(component, run_mode)

                    if self.should_populate(initialized_value):
                        value_dictionary = self.do_populate(initialized_value, run_mode, value_dictionary)

                    value_dictionary[component.identifier][run_mode] = initialized_value
            else:
                if "default" not in value_dictionary[component.identifier]:
                    initialized_value = self.initialize_value(component, "default")

                    if self.should_populate(initialized_value):
                        value_dictionary = self.do_populate(initialized_value, run_mode, value_dictionary)

                        # Ensure graph ref components are always unique to mode:
                        value_dictionary[component.identifier][run_mode] = initialized_value
                    else:
                        value_dictionary[component.identifier]["default"] = initialized_value

            for in_socket in list(component.in_sockets.values()):
                edge = in_socket.edge

                if edge is not None:
                    source_socket = edge.source_socket
                    activated_output_sockets.append((source_socket, run_mode))

        return value_dictionary

    def should_populate(self, value):
        for populate_key, populate_spec_dict in value.get_populate_items():
            if populate_key == "graph":
                return True

        return False

    def do_populate(self, value, mode, value_dictionary):
        for populate_key, populate_spec_dict in value.get_populate_items():
            if populate_key == "graph":
                spec = GraphSpecifications()
                spec.add_all(populate_spec_dict)
                graph = self.graph_repository.get(spec)[0]

                run = []
                for component_name, socket_name in value.get_required_graph_outputs():
                    run.append(graph.get_out_socket(component_name, socket_name))

                value_dictionary = self.build_value_dictionary([run], [mode], existing_dictionary=value_dictionary)

        return value_dictionary

    def list_unique_modes(self, component):
        required_modes = []

        for k, v in component.component_value.items():
            for variable in self.get_all_variables():
                ref = False
                for val in v:
                    if variable.referenced_in(val[0]):
                        ref = True
                        break
                if ref:
                    for mode in variable.defined_for():
                        if mode not in required_modes:
                            required_modes.append(mode)

        return required_modes

    def initialize_value(self, component, mode):
        updated_dict = {}

        for k,v in component.component_value.items():
            updated_dict[k] = v

            for variable in self.get_all_variables():
                updated_dict[k] = [(variable.replace_in_string(val[0], mode=mode), val[1]) for val in updated_dict[k]]

        value = component.component_type.initialize_value(updated_dict, component.language)
        value.set_component_name(component.name, mode)

        return value