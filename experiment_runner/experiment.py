class Experiment:

    model = None

    def train(self):
        loss = 0
        for iteration in self.max_iterations:
            loss += self.model.update()

            if iteration % self.report_every_n == 0:
                print(loss / self.report_every_n)
                loss = 0

            if iteration > 0 and iteration % self.validate_every_n:
                validation_evaluation = self.model.validate()
                print("Val: "+str(validation_evaluation))