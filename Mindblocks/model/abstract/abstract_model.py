class AbstractModel:

    identifier = None
    logger_manager = None

    def log(self, message, field, context):
        self.logger_manager.log(message, field, context)

    def get_identifier(self):
        return self.identifier