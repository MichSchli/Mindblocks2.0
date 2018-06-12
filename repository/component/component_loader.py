from repository.canvas.canvas_specifications import CanvasSpecifications
from repository.component.component_specifications import ComponentSpecifications


class ComponentLoader:

    def __init__(self, xml_helper, component_repository):
        self.xml_helper = xml_helper
        self.component_repository = component_repository

    def load_component(self, text, canvas_id, start_index):
        next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=start_index)

        if next_symbol != "component":
            print("ERROR")
            exit()

        component_specifications = ComponentSpecifications()
        component_specifications.canvas_id = canvas_id
        for key, value in attributes:
            component_specifications.add(key, value)

        component = self.component_repository.create(component_specifications)

        value_lines = {}
        current_value_line_id = None
        next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
        while next_symbol != "/component":
            if current_value_line_id is None:
                current_value_line_id = next_symbol
                if not current_value_line_id in value_lines:
                    value_lines[current_value_line_id] = []
            elif next_symbol == "/" + current_value_line_id:
                current_value_line_id = None
            else:
                value_lines[current_value_line_id].append((next_symbol, attributes))

            next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)

        component.value.load(value_lines)

        return component, pointer