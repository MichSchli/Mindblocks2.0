from Mindblocks.model.canvas.canvas_model import CanvasModel
from Mindblocks.repository.abstract.abstract_repository import AbstractRepository


class CanvasRepository(AbstractRepository):

    def __initialize_model__(self):
        return CanvasModel()

    def create(self, specifications):
        model = self.__create__()

        model.name = specifications.name

        return model