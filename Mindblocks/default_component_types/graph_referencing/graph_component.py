from Mindblocks.model.component_type.component_type_model import ComponentTypeModel
from Mindblocks.model.execution_graph.execution_component_value_model import ExecutionComponentValueModel


class GraphComponent(ComponentTypeModel):

    name = "GraphComponent"
    in_sockets = []
    out_sockets = []
    languages = ["python", "tensorflow"]

    def initialize_value(self, value_dictionary, language):
        value = GraphComponentValue()
        value.set_graph_name(value_dictionary["graph"][0][0])
        for in_link in value_dictionary["in_link"]:
            parts = in_link[0].split("->")
            value.add_in_link(parts[0], parts[1])

        for out_link in value_dictionary["out_link"]:
            parts = out_link[0].split("->")
            value.add_out_link(parts[1], parts[0])
        return value

    def execute(self, input_dictionary, value, output_models, mode):
        value.assign_input(input_dictionary)
        outputs = value.run_graph()

        for k,v in outputs.items():
            output_models[k].assign(v)

        return output_models

    def build_value_type_model(self, input_types, value, mode):
        value.assign_input_types(input_types)
        output_types = value.compute_types(mode)
        return output_types


class GraphComponentValue(ExecutionComponentValueModel):

    graph_name = None
    graph = None

    def set_graph(self, graph):
        self.graph = graph

    def __init__(self):
        self.in_links = []
        self.out_links = []

    def get_referenced_graphs(self):
        return [self.graph]

    def add_in_link(self, component_input, graph_input):
        self.in_links.append((component_input, graph_input))

    def add_out_link(self, component_output, graph_output):
        self.out_links.append((component_output, graph_output))

    def assign_input_types(self, input_dictionary):
        for component_input, graph_input in self.in_links:
            parts = graph_input.split(":")
            self.graph.enforce_type(parts[0], parts[1], input_dictionary[component_input])

    def assign_input(self, input_dictionary):
        for component_input, graph_input in self.in_links:
            parts = graph_input.split(":")
            self.graph.enforce_value(parts[0], parts[1], input_dictionary[component_input])

    def run_graph(self):
        results = self.graph.execute()
        return {output[0]: result for output, result in zip(self.out_links, results)}

    def compute_types(self, mode):
        results = self.graph.initialize_type_models(mode)
        return {output[0]: result for output, result in zip(self.out_links, results)}

    def set_graph_name(self, name):
        self.graph_name = name

    def get_populate_items(self):
        return [("graph", {"name": self.graph_name})]

    def get_required_graph_outputs(self):
        return [(l[1].split(":")[0], l[1].split(":")[1]) for l in self.out_links]

    def get_graph_inputs(self):
        return [(l[1].split(":")[0], l[1].split(":")[1]) for l in self.in_links]