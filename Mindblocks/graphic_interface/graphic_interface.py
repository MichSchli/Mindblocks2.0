import tkinter as tk

from Mindblocks.graphic_interface.menubar import Menubar
from Mindblocks.graphic_interface.toolbox import Toolbox


class GraphicInterface(tk.Tk):

    controller = None

    def __init__(self, controller):
        self.controller = controller
        tk.Tk.__init__(self)
        #Observable.__init__(self, events=['clicked', 'tab_changed'])

        '''
        Initialize controller
        '''

        '''
        Initialize selectors:
        '''
        #self.initialize_selectors()

        '''
        Initialize general UI:
        '''
        self.file_interface = None #FileInterface()
        self.title('Mindblocks')
        self.geometry('{}x{}'.format(800, 600))
        self.menubar = Menubar(self, self.file_interface)
        self.config(menu=self.menubar)
        self.make_support_frames()

        '''
        Initialize model-specific UI elements:
        '''

        self.initialize_toolbox(controller.get_component_type_repository())
        #self.initialize_description_panel()
        #self.initialize_canvas_area()

    def initialize_view(self):
        self.toolbox.initialize_view()

    def make_support_frames(self):
        self.left_frame = tk.Frame(self, background="blue")
        self.right_frame = tk.Frame(self, background="green")

        self.right_frame.pack(side=tk.RIGHT, expand=False, fill=tk.Y, pady=0, padx=0, anchor="ne")
        self.left_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, pady=0, padx=0, anchor="ne")


    def initialize_toolbox(self, component_type_repository):
        self.toolbox = Toolbox(self.right_frame, component_type_repository)
        self.toolbox.pack(side=tk.TOP, expand=True, fill=tk.Y, pady=0, padx=0, anchor="n")