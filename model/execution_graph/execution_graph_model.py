class ExecutionGraphModel:

    head_component = None
    components = None

    def __init__(self):
        self.components = []

    def add_head_component(self, head_component):
        self.head_component = head_component

    def execute(self):
        return self.head_component.pull()

    def count_components(self):
        return len(self.components)

    def get_components(self):
        return self.components

    def add_execution_component(self, execution_component):
        self.components.append(execution_component)