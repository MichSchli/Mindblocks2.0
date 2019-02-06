import tkinter as tk

class Menubar(tk.Menu):

    def get_menus(self):
        return [
            ("File",
             [
                 ("New", self.placeholder),
                 ("Save", self.placeholder),
                 ("Load", self.placeholder),
                 ("Exit", self.placeholder)
             ]
            ),
            ("Edit",
             [
                 ("Undo", self.placeholder),
                 ("Redo", self.placeholder),
                 ("Cut", self.placeholder),
                 ("Copy", self.placeholder),
                 ("Paste", self.placeholder)
             ]
            ),
            ("Canvas",
             [
                 ("Add Canvas", self.placeholder),
                 ("Save Canvas", self.placeholder),
                 ("Load Canvas", self.placeholder),
             ]
             )
        ]

    def __init__(self, root, file_interface):
        tk.Menu.__init__(self, root)
        self.root = root
        self.file_interface = file_interface
        
        for menu in self.get_menus():
            self.__add_menu_from_list__(*menu)

        events = []
        for root,options in self.get_menus():
            for option in options:
                events.append(root+":"+option[0])

    def placeholder(self, event_name):
        return lambda: print(event_name)
        
    def __add_menu_from_list__(self, root, options):
        menu = tk.Menu(self, tearoff=0)

        for option in options:
            event_name = root+":"+option[0]
            menu.add_command(label=option[0], command=option[1](event_name))

        self.add_cascade(label=root, menu=menu)
