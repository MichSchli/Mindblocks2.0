class CanvasModel:

    identifier = None
    name = None

    components = None

    def __init__(self):
        self.components = []

    def add_component(self, component):
        self.components.append(component)

    def get_components(self):
        return self.components

    def count_components(self):
        return len(self.components)