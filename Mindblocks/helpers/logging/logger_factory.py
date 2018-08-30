from Mindblocks.helpers.logging.logger import Logger

loggers = []

class LoggerFactory:

    def create_console_logger(self, console_configuration):
        new_logger = Logger(console_configuration, None)
        return new_logger