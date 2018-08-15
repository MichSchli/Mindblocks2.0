from Mindblocks.controller.graph_converter.tensorflow_section import TensorflowSection
from Mindblocks.model.execution_graph.execution_in_socket import ExecutionInSocket
from Mindblocks.model.execution_graph.execution_out_socket import ExecutionOutSocket
import tensorflow as tf


class TensorflowSectionContractor:

    def contract_tensorflow_sections_in_graphs(self, execution_graphs, run_modes):
        tensorflow_session = tf.Session()
        for execution_graph, mode in zip(execution_graphs, run_modes):
            self.contract_tensorflow_sections(execution_graph, mode, tensorflow_session)

        tensorflow_session.run(tf.global_variables_initializer())

    def contract_tensorflow_sections(self, execution_graph, mode, tensorflow_session):
        tensorflow_sections = self.find_tensorflow_sections(execution_graph)
        for tensorflow_section in tensorflow_sections:
            tensorflow_section.set_session(tensorflow_session)
            self.replace_tensorflow_section(execution_graph, tensorflow_section, mode)

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

    def replace_tensorflow_section(self, execution_graph, tensorflow_section, mode):
        execution_graph.add_execution_component(tensorflow_section)
        for component in tensorflow_section.components:
            for out_socket in component.get_out_sockets():
                target_in_sockets = out_socket.targets
                for in_socket in target_in_sockets:
                    if in_socket.execution_component.language != "tensorflow":
                        new_out_socket = ExecutionOutSocket()
                        in_socket.set_source(new_out_socket)
                        new_out_socket.execution_component = tensorflow_section

                        tensorflow_section.map_out_socket(out_socket, new_out_socket)

            for in_socket in component.get_in_sockets():
                source_out_socket = in_socket.source
                if source_out_socket.execution_component.language != "tensorflow":
                    new_in_socket = ExecutionInSocket()
                    source_out_socket.targets.remove(in_socket)
                    source_out_socket.add_target(new_in_socket)
                    new_in_socket.set_source(source_out_socket)
                    new_in_socket.cast = in_socket.cast
                    new_in_socket.execution_component = tensorflow_section

                    tensorflow_section.map_in_socket(in_socket, new_in_socket)

            execution_graph.components.remove(component)

        tensorflow_section.initialize_placeholders()
        tensorflow_section.compile(mode)