class MlHelper:

    evaluate_function = None
    update_and_loss_function = None
    loss_function = None

    def set_evaluate_function(self, execution_graph):
        self.evaluate_function = execution_graph

    def set_update_and_loss_function(self, run_graph):
        self.update_and_loss_function = run_graph

    def set_loss_function(self, execution_graph):
        self.loss_function = execution_graph

    def evaluate(self):
        self.evaluate_function.init_batches()
        performance = 0.0
        while self.evaluate_function.has_batches():
            performance += self.evaluate_function.execute()[0]
        return performance

    def train(self):
        for i in range(10000):
            self.update_and_loss_function.init_batches()
            while self.update_and_loss_function.has_batches():
                _, loss = self.update_and_loss_function.execute()

                print(loss)