from Mindblocks.helpers.logging.logger import Logger

loggers = []


class LoggerFactory:

    def setup(self, console_configuration, file_configuration):
        global loggers
        loggers = self.build(console_configuration, file_configuration)

    def build(self, console_configuration, file_configuration):
        new_loggers = [Logger(console_configuration, None)]
        return new_loggers

    def get(self):
        global loggers
        return loggers
