from Mindblocks.controller.ml_helper.ml_helper import MlHelper
from Mindblocks.controller.ml_helper.ml_helper_configuration import MlHelperConfiguration


class MlHelperFactory:

    graph_converter = None
    variable_repository = None
    logger_manager = None

    def __init__(self, graph_converter, variable_repository, tensorflow_session_repository, logger_manager):
        self.graph_converter = graph_converter
        self.variable_repository = variable_repository
        self.tensorflow_session_repository = tensorflow_session_repository
        self.logger_manager = logger_manager

    def build_configuration(self):
        configuration = MlHelperConfiguration()

        vars = self.variable_repository.get_by_name("max_iterations")
        if len(vars) > 0:
            configuration.max_iterations = int(vars[0].get_value())
        else:
            configuration.max_iterations = 500

        vars = self.variable_repository.get_by_name("report_loss_every_n")
        if len(vars) > 0:
            configuration.report_loss_every_n = int(vars[0].get_value())
        else:
            configuration.report_loss_every_n = None

        vars = self.variable_repository.get_by_name("validate_every_n")
        if len(vars) > 0:
            configuration.validate_every_n = int(vars[0].get_value())
        else:
            configuration.validate_every_n = 10

        configuration.report_perplexity = {"train": False,
                                           "validate": False,
                                           "test": False}
        vars = self.variable_repository.get_by_name("report_perplexity")
        if len(vars) > 0:
            for mode in ["train", "validate", "test"]:
                configuration.report_perplexity[mode] = vars[0].get_value(mode)

        return configuration

    def build_ml_helper_from_graph(self, graph, profile=False, log_dir=None):
        marked_sockets = graph.get_marked_sockets()

        update = marked_sockets["update"] if "update" in marked_sockets else None
        loss = marked_sockets["loss"] if "loss" in marked_sockets else None
        evaluate = marked_sockets["evaluate"] if "evaluate" in marked_sockets else None
        prediction = marked_sockets["prediction"] if "prediction" in marked_sockets else None

        return self.build_ml_helper(update=update,
                                    loss=loss,
                                    evaluate=evaluate,
                                    prediction=prediction,
                                    profile=profile,
                                    log_dir=log_dir)

    def build_ml_helper(self, update=None, loss=None, evaluate=None, prediction=None, profile=False, log_dir=None):
        ml_helper = MlHelper()
        tensorflow_session_model = self.tensorflow_session_repository.new()
        tensorflow_session_model.should_profile = profile
        ml_helper.set_tensorflow_session(tensorflow_session_model)
        ml_helper.profile_dir = log_dir

        runs = []
        run_interpretations = []
        run_modes = []

        if update is not None and loss is not None:
            update_and_loss_sockets = [update, loss]
            runs.append(update_and_loss_sockets)
            run_interpretations.append("update_and_loss")
            ml_helper.report_loss_after_updates = True
            run_modes.append("train")
        elif update is not None:
            update_sockets = [update, loss]
            runs.append(update_sockets)
            run_interpretations.append("update")
            ml_helper.report_loss_after_updates = False
            run_modes.append("train")
        if loss is not None:
            loss_socket = [loss]
            runs.append(loss_socket)
            run_interpretations.append("loss")
            run_modes.append("train")
        if evaluate is not None:
            evaluate_socket = [evaluate]
            runs.append(evaluate_socket)
            run_interpretations.append("evaluate")
            run_modes.append("test")

            evaluate_socket = [evaluate]
            runs.append(evaluate_socket)
            run_interpretations.append("validate")
            run_modes.append("validate")
        elif loss is not None:
            evaluate_socket = [loss]
            runs.append(evaluate_socket)
            run_interpretations.append("validate")
            run_modes.append("validate")
        if prediction is not None:
            prediction_socket = [prediction]
            runs.append(prediction_socket)
            run_interpretations.append("prediction")
            run_modes.append("test")

        run_graphs = self.graph_converter.to_executable(runs, run_modes=run_modes, tensorflow_session_model=tensorflow_session_model)

        for run_graph, interpretation in zip(run_graphs, run_interpretations):
            if interpretation == "update_and_loss":
                ml_helper.set_update_and_loss_function(run_graph)
            elif interpretation == "update":
                ml_helper.set_update_function(run_graph)
            elif interpretation == "loss":
                ml_helper.set_loss_function(run_graph)
            elif interpretation == "evaluate":
                ml_helper.set_evaluate_function(run_graph)
            elif interpretation == "validate":
                ml_helper.set_validate_function(run_graph)
            elif interpretation == "prediction":
                ml_helper.set_prediction_function(run_graph)

        ml_helper.configuration = self.build_configuration()
        ml_helper.logger_manager = self.logger_manager

        return ml_helper