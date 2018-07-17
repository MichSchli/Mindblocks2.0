from repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications


class ComponentLoader:

    def __init__(self, xml_helper, component_repository):
        self.xml_helper = xml_helper
        self.component_repository = component_repository

    def load_component(self, text, start_index):
        next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=start_index)

        if next_symbol != "component":
            print("ERROR")
            exit()

        component_specifications = CreationComponentSpecifications()
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
                    value_lines[current_value_line_id] = ""
            elif next_symbol == "/" + current_value_line_id:
                current_value_line_id = None
            else:
                value_lines[current_value_line_id] += next_symbol

            next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)

        for key, value in value_lines.items():
            component.set_attribute(key, value)

        return component, pointer