from view.headless_view.observers.input_text_observer import InputTextObserver


class RunGraphObserver(InputTextObserver):

    def accept(self, text):
        return text.startswith("run")

    def process(self, text):
        text = text.strip()
        parts = text.split(" ")

        specs = [item.split("=") for item in parts[2:]]
        spec_dict = {k:v for k,v in specs}

        if parts[1] == "graph":
            results = self.controller.run_graphs(spec_dict)

        for result in results:
            print(result)
