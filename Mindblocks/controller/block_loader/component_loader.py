from Mindblocks.model.creation_component.creation_component_in_socket import CreationComponentInSocket
from Mindblocks.model.creation_component.creation_component_out_socket import CreationComponentOutSocket
from Mindblocks.repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications


class ComponentLoader:

    def __init__(self, xml_helper, component_repository):
        self.xml_helper = xml_helper
        self.component_repository = component_repository

    def load_component(self, text, start_index=0, canvas_id=None, graph_id=None):
        next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=start_index)

        if next_symbol != "component":
            print("ERROR")
            exit()

        component_specifications = CreationComponentSpecifications()
        component_specifications.canvas_id = canvas_id
        component_specifications.graph_id = graph_id
        for key, value in attributes.items():
            component_specifications.add(key, value)

        component = self.component_repository.create(component_specifications)

        value_lines = {}
        current_value_line_id = None
        next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
        while next_symbol != "/component":
            if next_symbol == "mark":
                socket_name = attributes["socket"]
                next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
                component.place_mark(next_symbol, socket_name)
                next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
            elif next_symbol == "socket":
                socket_type = attributes["type"]
                next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
                if socket_type == "in":
                    in_socket = CreationComponentInSocket(component, next_symbol)
                    component.add_in_socket(in_socket)
                elif socket_type == "out":
                    out_socket = CreationComponentOutSocket(component, next_symbol)
                    component.add_out_socket(out_socket)
                next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
            elif current_value_line_id is None:
                current_value_line_id = next_symbol
                current_value_line_attributes = attributes
                if not current_value_line_id in value_lines:
                    value_lines[current_value_line_id] = []
            elif next_symbol == "/" + current_value_line_id:
                current_value_line_id = None
            else:
                value_lines[current_value_line_id] += [(next_symbol, current_value_line_attributes)]

            next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)

        for key, value in value_lines.items():
            component.set_attribute(key, value)

        return component, pointer