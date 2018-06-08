from repository.canvas.canvas_specifications import CanvasSpecifications


class CanvasLoader:

    def __init__(self, xml_helper, canvas_repository, component_loader):
        self.xml_helper = xml_helper
        self.component_loader = component_loader
        self.canvas_repository = canvas_repository

    def load_canvas(self, text, start_index):
        next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=start_index)

        if next_symbol != "canvas":
            print("ERROR")
            exit()

        canvas_specifications = CanvasSpecifications()
        for key, value in attributes:
            canvas_specifications.add(key, value)

        canvas = self.canvas_repository.create(canvas_specifications)

        while next_symbol != "/canvas":
            next_symbol = self.xml_helper.read_symbol(text, start_index=pointer)
            if next_symbol == "component":
                _, pointer = self.component_loader.load_component(text, canvas.identifier, start_index=pointer)
            else:
                _, _, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)

        return canvas, pointer