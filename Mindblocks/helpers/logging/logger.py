class Logger:

    filename = None

    def __init__(self, configuration, filename):
        self.configuration = configuration
        self.filename = filename

    def log(self, message, context, subcontext):
        if context in self.configuration:
            if subcontext in self.configuration[context] or "all" in self.configuration[context]:
                if self.filename is None:
                    print(message)
                else:
                    with open(self.filename, 'a') as log_file:
                        print(message, file=log_file)