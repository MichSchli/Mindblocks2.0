import unittest

from controller.controller import Controller


class TestCanvasController(unittest.TestCase):

    def test_add_canvas(self):
        controller = Controller()
        session_model = controller.initialize_model()
        controller.add_canvas()

        self.assertEqual(len(session_model.get_canvases()), 1)

    def test_add_multiple_canvases(self):
        controller = Controller()
        session_model = controller.initialize_model()
        controller.add_canvas()
        controller.add_canvas()

        self.assertEqual(len(session_model.get_canvases()), 2)

    def test_add_canvas_with_name(self):
        controller = Controller()
        session_model = controller.initialize_model()
        controller.add_canvas(specification_dict={"name":"test_name"})

        self.assertEqual(session_model.get_canvases()[0].get_name(), "test_name")

    def test_add_canvas_identifier(self):
        controller = Controller()
        session_model = controller.initialize_model()
        controller.add_canvas()
        controller.add_canvas()

        self.assertIsNotNone(session_model.get_canvases()[0].identifier)
        self.assertIsNotNone(session_model.get_canvases()[1].identifier)
        self.assertNotEqual(session_model.get_canvases()[0].identifier, session_model.get_canvases()[1].identifier)