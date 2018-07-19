class ExecutionHeadComponent:

    run_output_sockets = None

    def __init__(self, run_output_sockets):
        self.run_output_sockets = run_output_sockets

    def pull(self):
        return [socket.pull() for socket in self.run_output_sockets]