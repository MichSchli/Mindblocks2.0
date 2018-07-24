from controller.ml_helper.ml_helper import MlHelper


class MlHelperFactory:

    graph_converter = None

    def __init__(self, graph_converter):
        self.graph_converter = graph_converter

    def build_ml_helper(self, update=None, loss=None, evaluate=None, prediction=None):
        ml_helper = MlHelper()

        runs = []
        run_interpretations = []

        if update is not None and loss is not None:
            update_and_loss_sockets = [update, loss]
            runs.append(update_and_loss_sockets)
            run_interpretations.append("update_and_loss")
            ml_helper.report_loss_after_updates = True
        elif update is not None:
            update_sockets = [update, loss]
            runs.append(update_sockets)
            run_interpretations.append("update")
            ml_helper.report_loss_after_updates = False

        if loss is not None:
            loss_socket = [loss]
            runs.append(loss_socket)
            run_interpretations.append("loss")

        if evaluate is not None:
            evaluate_socket = [evaluate]
            runs.append(evaluate_socket)
            run_interpretations.append("evaluate")

        if prediction is not None:
            prediction_socket = [prediction]
            runs.append(prediction_socket)
            run_interpretations.append("prediction")

        run_graphs = self.graph_converter.to_executable(runs)

        for run_graph, interpretation in zip(run_graphs, run_interpretations):
            if interpretation == "update_and_loss":
                ml_helper.set_update_and_loss_function(run_graph)
            elif interpretation == "update":
                ml_helper.set_update_function(run_graph)
            elif interpretation == "loss":
                ml_helper.set_loss_function(run_graph)
            elif interpretation == "evaluate":
                ml_helper.set_evaluate_function(run_graph)
            elif interpretation == "prediction":
                ml_helper.set_prediction_function(run_graph)

        return ml_helper