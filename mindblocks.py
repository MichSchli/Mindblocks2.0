from controller.controller import Controller
from view.headless_view.headless_view import HeadlessView

controller = Controller()
controller.load_default_component_types()
session_model = controller.initialize_model()

controller.load_block_file("test_blocks/predict_iris_data.xml")
graph = controller.get_graphs()[0]

controller.train(graph)
controller.predict(graph)
controller.evaluate(graph)

#view = HeadlessView(session_model, controller)
#view.run()

