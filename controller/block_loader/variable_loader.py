from repository.creation_component_repository.creation_component_specifications import CreationComponentSpecifications
from repository.variable_repository.variable_specifications import VariableSpecifications


class VariableLoader:

    def __init__(self, xml_helper, variable_repository):
        self.xml_helper = xml_helper
        self.variable_repository = variable_repository

    def load_variable(self, text, start_index):
        next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=start_index)

        if next_symbol != "variable":
            print("ERROR")
            exit()

        variable_specifications = VariableSpecifications()
        for key, value in attributes.items():
            variable_specifications.add(key, value)

        variable = self.variable_repository.create(variable_specifications)

        while next_symbol != "/variable":
            next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
            if next_symbol == "default_value":
                next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
                variable.set_value(next_symbol)
                next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
            elif next_symbol.endswith("_value"):
                mode = next_symbol[:-6]
                next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)
                variable.set_value(next_symbol, mode=mode)
                next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)

        return variable, pointer