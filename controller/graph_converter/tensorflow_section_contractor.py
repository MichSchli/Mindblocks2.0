from controller.graph_converter.tensorflow_section import TensorflowSection


class TensorflowSectionContractor:

    def contract_tensorflow_sections(self, execution_graph):
        tensorflow_sections = self.find_tensorflow_sections(execution_graph)
        for tensorflow_section in tensorflow_sections:
            self.replace_tensorflow_section(execution_graph, tensorflow_section)

    def find_tensorflow_sections(self, execution_graph):
        tensorflow_section_map = {}

        for component in execution_graph.get_components():
            if component.identifier in tensorflow_section_map:
                pass
            elif component.language == "tensorflow":
                tensorflow_section = TensorflowSection()
                self.expand_tensorflow_section(component, tensorflow_section, tensorflow_section_map)

        return list(set(tensorflow_section_map.values()))

    def expand_tensorflow_section(self, component, tensorflow_section, tensorflow_section_map):
        tensorflow_section.add_component(component)
        tensorflow_section_map[component.identifier] = tensorflow_section

        for in_socket in component.get_in_sockets():
            if in_socket.source.execution_component.language == "tensorflow" and in_socket.source.execution_component.identifier not in tensorflow_section_map:
                self.expand_tensorflow_section(in_socket.source.execution_component, tensorflow_section, tensorflow_section_map)

        for out_socket in component.get_out_sockets():
            for target in out_socket.targets:
                if target.execution_component.language == "tensorflow" and target.execution_component.identifier not in tensorflow_section_map:
                    self.expand_tensorflow_section(target.execution_component, tensorflow_section, tensorflow_section_map)

    def replace_tensorflow_section(self, execution_graph, tensorflow_section):
        print(tensorflow_section.components)