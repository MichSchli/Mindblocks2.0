import tensorflow as tf

from Mindblocks.controller.ml_helper.initialization_helper import InitializationHelper
from Mindblocks.helpers.logging.logger_factory import LoggerFactory
import numpy as np

class MlHelper:
    evaluate_function = None
    update_and_loss_function = None
    loss_function = None
    validate_function = None
    prediction_function = None

    configuration = None

    model = None

    has_initialized = False

    tensorflow_session_model = None

    current_iteration = None

    profile_dir = None

    def __init__(self):
        self.best_validation_score = None
        self.best_epoch = None
        self.current_iteration = 0
        self.initialization_helper = InitializationHelper()

    def set_tensorflow_session(self, tensorflow_session_model):
        self.tensorflow_session_model = tensorflow_session_model

    def set_evaluate_function(self, execution_graph):
        self.evaluate_function = execution_graph

    def set_update_and_loss_function(self, run_graph):
        self.update_and_loss_function = run_graph

    def set_loss_function(self, execution_graph):
        self.loss_function = execution_graph

    def set_validate_function(self, execution_graph):
        self.validate_function = execution_graph

    def set_prediction_function(self, prediction_graph):
        self.prediction_function = prediction_graph

    def evaluate(self):
        self.initialize_model()
        self.evaluate_function.init_batches()
        performance = 0.0
        count = 0
        while self.evaluate_function.has_batches("test"):
            batch_result = self.evaluate_function.execute()[0]
            for b in batch_result:
                performance += b
                count += 1

        return  self.process_result_for_reporting(performance, count, mode="test")

    def validate(self):
        self.initialize_model()
        score = self.do_validate()
        return score

    def predict(self):
        self.initialize_model()
        self.prediction_function.init_batches()
        predictions = []

        while self.prediction_function.has_batches("test"):
            predictions.extend(self.prediction_function.execute()[0])

        return predictions

    def should_validate(self):
        return self.validate_function is not None

    def get_best_epoch(self):
        return self.best_epoch

    def get_best_validation_score(self):
        return self.best_validation_score

    def train(self, iterations=None):
        self.initialize_model()
        self.log("Starting training at iteration " + str(self.current_iteration), "training", "status")

        if iterations is None:
            iteration_range = range(self.current_iteration, self.configuration.max_iterations)
        else:
            iteration_range = range(self.current_iteration, self.current_iteration + iterations)

        self.first_batch = True
        for i in iteration_range:
            self.log("Starting iteration " + str(i), "training", "iteration")
            self.current_iteration = i
            self.tensorflow_session_model.update_iteration(i)
            self.do_train_iteration()

            if self.should_validate() and i % self.configuration.validate_every_n == 0:
                validation_performance = self.do_validate()
                message, context, field = "Validation at epoch " + str(i) + ": " + str(
                    validation_performance), "validation", "performance"
                self.log(message, context, field)

        self.current_iteration += 1

    def log(self, message, context, field):
        self.logger_manager.log(message, context, field)

    def save(self, filepath):
        self.tensorflow_session_model.save(filepath + "/tfparams.ckpt")

    def load(self, filepath):
        self.tensorflow_session_model.load(filepath + "/tfparams.ckpt")

    def initialize_model(self):
        if not self.has_initialized:
            self.log("Initializing model.", "training", "status")
            self.initialization_helper.initialize([self.update_and_loss_function,
                                                   self.prediction_function,
                                                   self.validate_function,
                                                   self.evaluate_function])
            self.has_initialized = True
            self.count_parameters()

    def do_validate(self):
        self.validate_function.init_batches()
        performance = 0.0
        count = 0
        while self.validate_function.has_batches("validate"):
            batch_result = self.validate_function.execute()[0]
            for b in batch_result:
                performance += b
                count += 1

        score = self.process_result_for_reporting(performance, count, mode="validate")

        if self.best_validation_score is None \
                or (self.best_validation_score > score and self.configuration.minimize_validation_score) \
                or (self.best_validation_score < score and not self.configuration.minimize_validation_score):
            self.best_validation_score = score
            self.best_epoch = self.current_iteration

        return score

    def process_result_for_reporting(self, performance, count, mode):
        average_performance = performance / count

        if self.configuration.report_perplexity[mode]:
            perplexity = np.exp(average_performance)
            return perplexity

        return average_performance

    def count_parameters(self):
        message = "Learnable parameter count: "
        context = "training"
        field = "parameters"
        self.log(message, context, field)

        if self.update_and_loss_function is not None:
            parameters = self.update_and_loss_function.count_parameters()
        else:
            parameters = 0
            self.log("No learnable parameters as no update graph has been defined", context, field)

        message = " * Total: " + str(parameters)
        self.log(message, context, field)

        return parameters

    def do_train_iteration(self):
        self.update_and_loss_function.init_batches()
        batch = 1

        loss_tracker = 0
        count = 0
        while self.update_and_loss_function.has_batches("train"):
            _, loss = self.update_and_loss_function.execute()

            if self.first_batch and self.profile_dir is not None:
                self.first_batch = False
                self.tensorflow_session_model.generate_profiling_data(self.profile_dir)

            for b in loss:
                loss_tracker += b
                count += 1

            if self.configuration.report_loss_every_n is not None and batch % self.configuration.report_loss_every_n == 0:
                out_loss = self.process_result_for_reporting(loss_tracker, count, mode="train")
                message = "Loss at batch " + str(batch) + ": " + str(out_loss)
                context = "training"
                field = "loss"
                self.log(message, context, field)
                loss_tracker = 0
                count = 0

            batch += 1
