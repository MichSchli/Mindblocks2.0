class ExecutionGraphModel:

    head_component = None
    components = None

    def __init__(self):
        self.components = []

    def add_head_component(self, head_component):
        self.head_component = head_component

    def execute(self):
        self.clear_all_caches()
        return self.head_component.pull()

    def clear_all_caches(self):
        self.head_component.clear_caches()

    def count_components(self):
        return len(self.components)

    def get_components(self):
        return self.components

    def add_execution_component(self, execution_component):
        self.components.append(execution_component)

    def init_batches(self):
        for component in self.components:
            if component.execution_type is not None and component.execution_type.name == "BatchGenerator":
                component.execution_value.init_batches()

    def has_batches(self):
        for component in self.components:
            if component.execution_type is not None and component.execution_type.name == "BatchGenerator":
                return component.execution_value.has_unyielded_batches()