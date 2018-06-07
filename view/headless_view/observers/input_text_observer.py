class InputTextObserver:

    controller = None

    def __init__(self, controller):
        self.controller = controller

    def accept(self, text):
        pass

    def process(self, text):
        pass

    def notify(self, message_dictionary):
        if self.accept(message_dictionary["text"]):
            self.process(message_dictionary["text"])