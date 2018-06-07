import uuid


class IdentifierRepository:

    def create(self):
        return uuid.uuid4()