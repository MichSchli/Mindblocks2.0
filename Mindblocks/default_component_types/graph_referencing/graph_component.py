from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class GraphComponent(ComponentTypeModel):

    name = "GraphComponent"
    in_sockets = []
    out_sockets = []
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary):
        value = GraphComponentValue()
        value.set_graph_name(value_dictionary["graph"][0])
        for in_link in value_dictionary["in_link"]:
            parts = in_link.split("->")
            value.add_in_link(parts[0], parts[1])

        for out_link in value_dictionary["out_link"]:
            parts = out_link.split("->")
            value.add_out_link(parts[1], parts[0])
        return value

    def execute(self, input_dictionary, value, mode):
        value.assign_input(input_dictionary, mode)
        output = value.run_graph(mode)
        return output

    def infer_types(self, input_types, value):
        return {"output": input_types["input"]}

    def infer_dims(self, input_dims, value):
        return {"output": input_dims["input"]}


class GraphComponentValue(ExecutionComponentValueModel):

    graph_name = None
    graph = None

    def __init__(self):
        self.in_links = []
        self.out_links = []

    def add_in_link(self, component_input, graph_input):
        self.in_links.append((component_input, graph_input))

    def add_out_link(self, component_output, graph_output):
        self.out_links.append((component_output, graph_output))

    def assign_input(self, input_dictionary, mode):
        for component_input, graph_input in self.in_links:
            parts = graph_input.split(":")
            self.graph[mode].enforce_value(parts[0], parts[1], input_dictionary[component_input])

    def run_graph(self, mode):
        results = self.graph[mode].execute()
        return {output[0]: result for output, result in zip(self.out_links, results)}


    def set_graph_name(self, name):
        self.graph_name = name

    def get_populate_items(self):
        return [("graph", {"name": self.graph_name})]

    def get_required_graph_outputs(self):
        return [(l[1].split(":")[0], l[1].split(":")[1]) for l in self.out_links]