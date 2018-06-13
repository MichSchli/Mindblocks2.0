from experiment_runner.experiment import Experiment
from experiment_runner.experiment_preprocessor import ExperimentPreprocessor


class ExperimentBuilder:

    def __init__(self):
        self.experiment_preprocessor = ExperimentPreprocessor()

    def build_experiment(self, graph):
        experiment = Experiment()
        experiment.model = self.experiment_preprocessor.build_model(graph)
        return experiment