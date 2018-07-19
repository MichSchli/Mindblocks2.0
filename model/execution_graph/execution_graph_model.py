class ExecutionGraphModel:

    head_component = None

    def add_head_component(self, head_component):
        self.head_component = head_component

    def execute(self):
        return self.head_component.pull()