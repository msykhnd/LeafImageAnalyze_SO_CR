import tkinter
from gui import app

root = tkinter.Tk()
root.title("Leaf Image")
root.option_add('*font', ('MS Sans Serif', 16))
ap = app.AppForm(master=root)
ap.mainloop()
