import tensorflow as tf


class InitializationHelper:

    def initialize(self, run_graphs):
        session = None
        for run in run_graphs:
            if run is not None:
                run.initialize()

            if run is not None and run.tensorflow_session is not None:
                session = run.tensorflow_session

        if session is not None:
            session.run(tf.global_variables_initializer())