class AbstractModel:

    logger_manager = None

    def log(self, message, field, context):
        self.logger_manager.log(message, field, context)