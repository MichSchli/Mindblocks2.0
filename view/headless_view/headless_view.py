from view.headless_view.observers.add_element_observer import AddElementObserver
from view.headless_view.observers.describe_element_observer import DescribeElementObserver
from view.headless_view.observers.list_element_observer import ListElementObserver
from view.headless_view.observers.load_file_observer import LoadFileObserver
from view.headless_view.observers.run_graph_observer import RunGraphObserver


class HeadlessView:

    session_model = None

    def __init__(self, session_model, controller):
        self.session_model = session_model

        self.observers = [AddElementObserver(controller),
                          ListElementObserver(controller),
                          LoadFileObserver(controller),
                          DescribeElementObserver(controller),
                          RunGraphObserver(controller)]

    def run(self):
        while True:
            user_input = input("Enter next command:")

            self.process(user_input)

    def process(self, user_input):
        if user_input == "exit":
            exit()
        else:
            for observer in self.observers:
                observer.notify({"type": "input_text",
                                 "text": user_input})