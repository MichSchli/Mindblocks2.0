from view.headless_view.observers.input_text_observer import InputTextObserver


class ListElementObserver(InputTextObserver):

    def accept(self, text):
        return text.startswith("list")

    def process(self, text):
        text = text.strip()
        parts = text.split(" ")

        specs = [item.split("=") for item in parts[2:]]
        spec_dict = {k:v for k,v in specs}

        if parts[1] == "canvas":
            elements = self.controller.get_canvases(spec_dict)
        elif parts[1] == "component":
            elements = self.controller.get_components(spec_dict)
        elif parts[1] == "graph":
            elements = self.controller.get_graphs(spec_dict)

        for element in elements:
            print(element)
