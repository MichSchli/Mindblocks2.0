from Mindblocks.model.canvas.canvas_model import CanvasModel
from Mindblocks.model.tensorflow_session.tensorflow_session_model import TensorflowSessionModel
from Mindblocks.repository.abstract.abstract_repository import AbstractRepository
import tensorflow as tf

from Mindblocks.repository.tensorflow_session_repository.tensorflow_session_specifications import \
    TensorflowSessionSpecifications


class TensorflowSessionRepository(AbstractRepository):

    def __initialize_model__(self):
        return TensorflowSessionModel()

    def get_specifications(self):
        return TensorflowSessionSpecifications()

    def create(self, specifications):
        model = self.__create__()
        model.tensorflow_session = tf.Session()

        return model