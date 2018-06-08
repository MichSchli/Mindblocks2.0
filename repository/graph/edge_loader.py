from repository.component.component_specifications import ComponentSpecifications


class EdgeLoader:

    def __init__(self, xml_helper, component_repository, graph_repository):
        self.xml_helper = xml_helper
        self.component_repository = component_repository
        self.graph_repository = graph_repository

    def load_edge(self, text, pointer):
        next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
        next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)

        source = None
        source_socket_id = None
        target = None
        target_socket_id = None

        expect_source_socket = False
        expect_target_socket = False
        while next_symbol != "/edge":
            if next_symbol == "source_socket":
                component_name = dict(attributes)["component_name"]
                component_spec = ComponentSpecifications()
                component_spec.name = component_name

                source = self.component_repository.get(component_spec)[0]
                expect_source_socket = True
            elif expect_source_socket:
                source_socket_id = source.component_type.get_out_socket_id(next_symbol)
                expect_source_socket = False
            if next_symbol == "target_socket":
                component_name = dict(attributes)["component_name"]
                component_spec = ComponentSpecifications()
                component_spec.name = component_name

                target = self.component_repository.get(component_spec)[0]
                expect_target_socket = True
            elif expect_target_socket:
                target_socket_id = target.component_type.get_in_socket_id(next_symbol)
                expect_target_socket = False

            next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)

        self.graph_repository.create_edge(source, source_socket_id, target, target_socket_id)

        return None, pointer
