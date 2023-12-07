from tkinter import *

from neural_network import NeuralNetwork


class Window:
    def __init__(self, neural_network):
        self.root = Tk()
        self.columns = 5
        self.rows = 5
        self.neural_network = neural_network

    def create_new_window(self):
        self.root.title("Testing ML")
        self.root.geometry("800x600")
        img_label = Label(self.root)
        img_label.grid(row=0, column=0, columnspan=self.columns)
        self.generate_button()
        self.root.mainloop()

    def generate_button(self):
        g_b = Button(self.root, text="Generate image", bg="#54FA9B")
        g_b.place(x=50, y=50)
