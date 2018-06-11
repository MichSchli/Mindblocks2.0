class Visitor:

    def run_visit(self, graph, function, arguments={}):
        outputs = []
        for vertex in graph.topological_walk():
            vertex_output = function(vertex, arguments=arguments)
            outputs.extend(vertex_output)
        return vertex_output

    def yield_visit(self, graph, function, arguments={}):
        for vertex in graph.topological_walk():
            for line in function(vertex, arguments=arguments):
                yield line

