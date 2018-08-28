import tensorflow as tf

from Mindblocks.controller.ml_helper.initialization_helper import InitializationHelper
from Mindblocks.helpers.logging.logger_factory import LoggerFactory


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
        self.current_iteration = 0
        self.initialization_helper = InitializationHelper()
        self.logger_factory = LoggerFactory()

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
        while self.evaluate_function.has_batches():
            performance += self.evaluate_function.execute()[0]
        return performance

    def validate(self):
        self.initialize_model()
        return self.do_validate()

    def predict(self):
        self.initialize_model()
        self.prediction_function.init_batches()
        predictions = []

        while self.prediction_function.has_batches():
            predictions.extend(self.prediction_function.execute()[0])

        return predictions

    def should_validate(self):
        return self.validate_function is not None

    def train(self, iterations=None):
        self.log("Starting training at iteration " + str(self.current_iteration), "training", "status")
        self.initialize_model()

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
        loggers = self.logger_factory.get()
        for logger in loggers:
            logger.log(message, context, field)

    def initialize_model(self):
        if not self.has_initialized:
            self.log("Initializing model.", "initialization", "status")
            self.initialization_helper.initialize([self.update_and_loss_function,
                                                   self.prediction_function,
                                                   self.validate_function,
                                                   self.evaluate_function])
            self.has_initialized = True

    def do_validate(self):
        self.validate_function.init_batches()
        performance = 0.0
        while self.validate_function.has_batches():
            performance += self.validate_function.execute()[0]
        return performance

    def do_train_iteration(self):
        self.update_and_loss_function.init_batches()
        batch = 1
        loss_tracker = 0
        while self.update_and_loss_function.has_batches():
            _, loss = self.update_and_loss_function.execute()

            if self.first_batch and self.profile_dir is not None:
                self.first_batch = False
                self.tensorflow_session_model.generate_profiling_data(self.profile_dir)

            loss_tracker += loss

            if self.configuration.report_loss_every_n is not None and batch % self.configuration.report_loss_every_n == 0:
                out_loss = loss_tracker / self.configuration.report_loss_every_n
                message = "Loss at batch " + str(batch) + ": " + str(out_loss)
                context = "training"
                field = "loss"
                self.log(message, context, field)
                loss_tracker = 0

            batch += 1
