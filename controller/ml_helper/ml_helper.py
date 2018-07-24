class MlHelper:

    evaluate_function = None

    def set_evaluate_function(self, execution_graph):
        self.evaluate_function = execution_graph

    def evaluate(self):
        self.evaluate_function.init_batches()
        performance = self.evaluate_function.execute()[0]
        return performance