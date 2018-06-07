from model.canvas.canvas_model import CanvasModel
from repository.abstract_repository import AbstractRepository


class CanvasRepository(AbstractRepository):

    def create(self, specifications):
        canvas_model = CanvasModel()
        canvas_model.name = specifications.name
        canvas_model.identifier = self.identifier_repository.create()

        self.__create__(canvas_model)

        return canvas_model