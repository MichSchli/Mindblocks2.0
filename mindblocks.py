from controller.controller import Controller
from view.headless_view.headless_view import HeadlessView

controller = Controller()
session_model = controller.initialize_model()
view = HeadlessView(session_model, controller)
view.run()