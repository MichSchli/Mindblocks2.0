class ConfigurationLoader:

    def __init__(self, xml_helper, variable_loader):
        self.xml_helper = xml_helper
        self.variable_loader = variable_loader

    def load_configuration(self, text, start_index):
        next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=start_index)

        if next_symbol != "configuration":
            print("ERROR")
            exit()

        while next_symbol != "/configuration":
            next_symbol = self.xml_helper.read_symbol(text, start_index=pointer)
            if next_symbol == "variable":
                _, pointer = self.variable_loader.load_variable(text, start_index=pointer)
            else:
                _, _, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)

        return None, pointer