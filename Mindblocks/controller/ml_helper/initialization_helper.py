import tensorflow as tf


class InitializationHelper:

    def initialize(self, run_graphs):
        session_model = None
        for run in run_graphs:
            if run is not None:
                print("======")
                run.initialize()

            if run is not None and run.tensorflow_session_model is not None:
                session_model = run.tensorflow_session_model

        if session_model is not None:
            session_model.initialize_variables()