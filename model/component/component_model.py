class ComponentModel:

    name = None
    identifier = None
    canvas_id = None
    canvas_name = None
    graph_id = None

    component_type = None

    value = None

    def get_name(self):
        return self.name

    def run_python(self):
        self.component_type.execute(self.value)

    def __str__(self):
        return "Component: "+str(self.name)+ " (" + (str(self.component_type.name) if self.component_type is not None else "None") + ") |"+str(self.identifier) +" canvas_name="+str(self.canvas_name)

    def describe(self):
        description = "Component: "+str(self.name)+ " (" + (str(self.component_type.name) if self.component_type is not None else "None") + ") |"+str(self.identifier) + "\n"
        description += "canvas_name=" + str(self.canvas_name) + "\n"
        description += "graph_id=" + str(self.graph_id) + "\n"
        if self.value != None:
            description += self.value.describe() + "\n"

        return description