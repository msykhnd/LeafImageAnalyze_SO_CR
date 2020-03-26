import tkinter
from gui.app import AppForm


root = tkinter.Tk()
root.title("Leaf Image")
root.option_add('*font', ('MS Sans Serif', 16))
app = AppForm(master=root)
app.mainloop()
