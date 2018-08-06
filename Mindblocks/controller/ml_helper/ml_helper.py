class MlHelper:

    evaluate_function = None
    update_and_loss_function = None
    loss_function = None
    validate_function = None

    configuration = None

    def set_evaluate_function(self, execution_graph):
        self.evaluate_function = execution_graph

    def set_update_and_loss_function(self, run_graph):
        self.update_and_loss_function = run_graph

    def set_loss_function(self, execution_graph):
        self.loss_function = execution_graph

    def set_validate_function(self, execution_graph):
        self.validate_function = execution_graph

    def evaluate(self):
        self.evaluate_function.init_batches()
        performance = 0.0
        while self.evaluate_function.has_batches():
            performance += self.evaluate_function.execute()[0]
        return performance

    def train(self):
        for i in range(self.configuration.max_iterations):
            self.do_train_iteration()

            if i % self.configuration.validate_every_n == 0:
                print(self.do_validate())

    def do_validate(self):
        self.validate_function.init_batches()
        performance = 0.0
        while self.validate_function.has_batches():
            performance += self.validate_function.execute()[0]
        return performance

    def do_train_iteration(self):
        self.update_and_loss_function.init_batches()
        batch = 0
        loss_tracker = 0
        while self.update_and_loss_function.has_batches():
            _, loss = self.update_and_loss_function.execute()

            loss_tracker += loss

            if self.configuration.report_loss_every_n is not None and batch % self.configuration.report_loss_every_n == 0:
                print(loss_tracker / self.configuration.report_loss_every_n)
                loss_tracker = 0

            batch += 1