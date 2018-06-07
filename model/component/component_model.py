class ComponentModel:

    name = None
    identifier = None
    canvas_id = None
    canvas_name = None
    graph_id = None

    component_type = None

    def get_name(self):
        return self.name

    def __str__(self):
        return "Component: "+str(self.name)+"|"+str(self.identifier) +" canvas_name="+str(self.canvas_name)