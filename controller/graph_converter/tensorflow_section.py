class TensorflowSection:

    components = None

    def __init__(self):
        self.components = []

    def add_component(self, component):
        self.components.append(component)