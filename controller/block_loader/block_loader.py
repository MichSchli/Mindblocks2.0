class BlockLoader:

    def __init__(self, xml_helper, canvas_loader):
        self.xml_helper = xml_helper
        self.canvas_loader = canvas_loader

    def load(self, filename):
        file_lines = ""
        block_file = open(filename, 'r')
        for line in block_file:
            line = line.strip()
            file_lines += line
        block_file.close()

        pointer = 0
        next_symbol = self.xml_helper.read_symbol(file_lines, start_index=pointer)
        while next_symbol is not None:
            if next_symbol == "block" or next_symbol == "/block":
                _,_,pointer = self.xml_helper.pop_symbol(file_lines, start_index=pointer)
            elif next_symbol == "canvas":
                _, pointer = self.canvas_loader.load_canvas(file_lines, start_index=pointer)
            else:
                # TODO proper error handling
                print("ERROR: UNKNOWN SYMBOL")
                exit()

            next_symbol = self.xml_helper.read_symbol(file_lines, start_index=pointer)