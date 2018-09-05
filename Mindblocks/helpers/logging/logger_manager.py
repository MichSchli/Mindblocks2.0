class LoggerManager:

    loggers = None
    factory = None

    def __init__(self, factory):
        self.factory = factory
        self.loggers = []

    def add_console_logger(self, logger_configuration):
        logger = self.factory.create_console_logger(logger_configuration)
        self.loggers.append(logger)

    def add_file_logger(self, logger_configuration, log_file_name):
        logger = self.factory.create_file_logger(logger_configuration, log_file_name)
        self.loggers.append(logger)

    def add_default_console_logger(self):
        default_config = {"training": ["status",
                                       "loss",
                                       "parameters"],
                          "validation": ["all"],
                          "batching": ["status"]}
        self.add_console_logger(default_config)

    def log(self, message, context, field):
        for logger in self.loggers:
            logger.log(message, context, field)