from Mindblocks.repository.canvas_repository.canvas_specifications import CanvasSpecifications
from Mindblocks.repository.graph.graph_specifications import GraphSpecifications


class GraphLoader:

    def __init__(self, xml_helper, component_loader, edge_loader, graph_repository):
        self.xml_helper = xml_helper
        self.component_loader = component_loader
        self.graph_repository = graph_repository
        self.edge_loader = edge_loader

    def load_graph(self, text, start_index):
        next_symbol, attributes, pointer = self.xml_helper.pop_symbol(text, start_index=start_index)

        if next_symbol != "graph":
            print("ERROR")
            exit()

        graph_specifications = GraphSpecifications()
        for key, value in attributes.items():
            graph_specifications.add(key, value)

        graph = self.graph_repository.create(graph_specifications)

        while next_symbol != "/graph":
            next_symbol = self.xml_helper.read_symbol(text, start_index=pointer)
            if next_symbol == "component":
                _, pointer = self.component_loader.load_component(text, graph_id=graph.identifier, start_index=pointer)
            elif next_symbol == "edge":
                _, pointer = self.edge_loader.load_edge(text, pointer, graph_id=graph.identifier)
            else:
                _, _, pointer = self.xml_helper.pop_symbol(text, start_index=pointer)

        return graph, pointer