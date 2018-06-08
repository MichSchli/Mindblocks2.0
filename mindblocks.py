from controller.controller import Controller
from view.headless_view.headless_view import HeadlessView

controller = Controller()
controller.load_default_component_types()
session_model = controller.initialize_model()

controller.load_block_file("test_blocks/add_constants.xml")
controller.run_graphs()

view = HeadlessView(session_model, controller)
view.run()

