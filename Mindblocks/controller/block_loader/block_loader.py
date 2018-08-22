from Mindblocks.error_handling.loading.component_not_found_exception import ComponentNotFoundException
from Mindblocks.error_handling.loading.socket_not_found_exception import SocketNotFoundException


class BlockLoader:

    def __init__(self, xml_helper, canvas_loader, configuration_loader):
        self.xml_helper = xml_helper
        self.canvas_loader = canvas_loader
        self.configuration_loader = configuration_loader

    def load(self, filename):
        file_lines = ""
        block_file = open(filename, 'r')
        for line in block_file:
            line = line.strip()
            file_lines += line
        block_file.close()

        self.load_block(file_lines)

    def load_block(self, file_lines, start_index=0):
        pointer = start_index
        next_symbol = self.xml_helper.read_symbol(file_lines, start_index=pointer)
        while next_symbol is not None:
            if next_symbol == "block" or next_symbol == "/block":
                _, _, pointer = self.xml_helper.pop_symbol(file_lines, start_index=pointer)
            elif next_symbol == "canvas":
                try:
                    _, pointer = self.canvas_loader.load_canvas(file_lines, start_index=pointer)
                except SocketNotFoundException:
                    raise
                except ComponentNotFoundException:
                    raise
            elif next_symbol == "configuration":
                _, pointer = self.configuration_loader.load_configuration(file_lines, start_index=pointer)
            else:
                _, _, pointer = self.xml_helper.pop_symbol(file_lines, start_index=pointer)

            next_symbol = self.xml_helper.read_symbol(file_lines, start_index=pointer)