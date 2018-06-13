from experiment_runner.model.model import Model


class ExperimentPreprocessor:

    def __init__(self):
        pass

    def build_model(self, graph):
        model = Model()
        graph.initialize()

        ml_trainer = graph.get_components("MlTrainer")[0]
        model.update_graph = graph.build_partial_copy(ml_trainer)
        return model


        #ml_tester = graph.get_components("MlPredictor")[0]
        #ml_tester = graph.get_components("MlTester")[0]
