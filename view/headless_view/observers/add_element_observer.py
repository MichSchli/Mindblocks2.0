from view.headless_view.observers.input_text_observer import InputTextObserver


class AddElementObserver(InputTextObserver):

    def accept(self, text):
        return text.startswith("add")

    def process(self, text):
        text = text.strip()
        parts = text.split(" ")

        specs = [item.split("=") for item in parts[2:]]
        spec_dict = {k:v for k,v in specs}

        if parts[1] == "canvas":
            self.controller.add_canvas(spec_dict)
        elif parts[1] == "component":
            self.controller.add_component(spec_dict)
        elif parts[1] == "graph":
            self.controller.add_graph(spec_dict)