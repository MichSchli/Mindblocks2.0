from model.graph.graph_runners.visitor import Visitor


class GraphRunner(Visitor):

    def __run_vertex__(self, vertex, arguments={}):
        return vertex.run()

    def run(self, graph, input_dictionary):
        output_dictionary = self.run_visit(graph, self.__run_vertex__)
        return output_dictionary