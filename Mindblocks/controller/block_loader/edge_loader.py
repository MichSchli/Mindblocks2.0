from Mindblocks.repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications


class EdgeLoader:

    def __init__(self, xml_helper, graph_repository, component_repository):
        self.xml_helper = xml_helper
        self.graph_repository = graph_repository
        self.component_repository = component_repository

    def load_edge(self, text, start_index, graph_id=None):
        next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=start_index)

        if next_symbol != "edge":
            print("ERROR")
            exit()

        source_socket = None
        target_socket = None

        cast = None
        if "cast" in attributes:
            cast = attributes["cast"]

        while next_symbol != "/edge":
            next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
            if next_symbol == "source":
                socket_name = attributes["socket"]
                next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)

                component_name = next_symbol
                component_spec = CreationComponentSpecifications()
                component_spec.graph_id = graph_id
                component_spec.name = component_name
                component = self.component_repository.get(component_spec)[0]

                source_socket = component.get_out_socket(socket_name)

                next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
            elif next_symbol == "target":
                socket_name = attributes["socket"]
                next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)

                component_name = next_symbol
                component_spec = CreationComponentSpecifications()
                component_spec.graph_id = graph_id
                component_spec.name = component_name
                component = self.component_repository.get(component_spec)[0]

                target_socket = component.get_in_socket(socket_name)

                next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)

        edge = self.graph_repository.add_edge(source_socket, target_socket, cast=cast)

        return edge, pointer