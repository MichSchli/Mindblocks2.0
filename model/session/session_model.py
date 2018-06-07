class SessionModel:

    canvases = None

    def __init__(self):
        self.canvases = []

    def get_canvases(self):
        return self.canvases

    def add_canvas(self, canvas):
        self.canvases.append(canvas)