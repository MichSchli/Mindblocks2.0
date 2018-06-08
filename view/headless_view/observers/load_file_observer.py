from view.headless_view.observers.input_text_observer import InputTextObserver


class LoadFileObserver(InputTextObserver):

    def accept(self, text):
        return text.startswith("load")

    def process(self, text):
        text = text.strip()
        parts = text.split(" ")

        filename = parts[1]
        self.controller.load_block_file(filename)