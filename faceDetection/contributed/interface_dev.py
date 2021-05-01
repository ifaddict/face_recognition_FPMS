import tkinter as tk
from tkinter import *
from tkinter import ttk
import cv2
from PIL import Image
from PIL import ImageTk

class NewWindow(tk.Toplevel):

    def __init__(self, master=None):
        super().__init__(master=master)
        self.title("Options")
        self.geometry("300x300")
        label = tk.Label(self, text="Options Window")
        label.pack()


        # Creating a Option Menu for objetCONF
        # Set the variable for objetCONF and create the list
        # of options by initializing the constructor
        # of class OptionMenu.
        objetCONF_Variable = tk.StringVar(self)
        objetCONF_Variable.set("5 (actuel)")
        objetCONF_Option = tk.OptionMenu(self,
                                      objetCONF_Variable,
                                      "0.2", "0.3", "0.4 (recommandé)", "0.5")
        objetCONF_Option.pack()

        # Creating a Option Menu for visageCONF
        # Set the variable for visageCONF and create the list
        # of options by initializing the constructor
        # of class OptionMenu.
        visageCONF_Variable = tk.StringVar(self)
        visageCONF_Variable.set("2 (actuel)")
        visageCONF_Option = tk.OptionMenu(self, visageCONF_Variable,
                                     "0.5", "0.6", "0.7 (recommandé)",
                                     "0.8")
        visageCONF_Option.pack()



        btn_create = tk.ttk.Button(self, text='Enregistrer', width=34)
        btn_create.place(x=50, y=200, height=35)



def mainWindow():
    root = tk.Tk()

    root.wm_iconbitmap('@fpms.xbm')
    root.title("EDGE IA")
    root.geometry('1040x480')
    stateText = ""

    cap = cv2.VideoCapture(0)

    # On lit la première frame
    ret, frame = cap.read()

    # ►►►► GUI ◄◄◄◄

    # On passe par Pillow pour avoir des images sous Tkinter
    image = Image.fromarray(frame)
    photo = ImageTk.PhotoImage(image)

    # Création d'un canvas pour y afficher le flux vidéo
    canvas = tk.Canvas(root, width=photo.width() - 10, height=photo.height() - 10, bd=3, relief='ridge')
    canvas.pack(side='left', fill='both', expand=True)

    canvas.create_image((0, 0), image=photo, anchor='nw')

    # Création d'un canvas pour y afficher le reste de l'interface
    canvas2 = tk.Canvas(root, width=400, height=455, bg="#19647E")
    canvas2.pack(anchor='ne', fill='both', expand=True)

    # imgFpms sera le background de l'interface


    info = tk.Label(root, text="Visage inconnu ? \n Choisissez un nom et \n placez-vous devant la \n caméra",
                    bg='steel blue', fg='black', font=('times', 15, 'bold'))
    info.place(x=660, y=30)

    label = tk.Label(root, text="Label : ", bg="#7a86ac")
    label.place(x=660, y=200)

    entryLabel = tk.ttk.Entry(root, width=35)
    entryLabel.place(x=730, y=200)

    code = tk.Label(root, text="Code : ", bg="#7a86ac")
    code.place(x=660, y=230)
    entryCode = tk.ttk.Entry(root, width=35)
    entryCode.place(x=730, y=230)





    click_btn = PhotoImage(file='icons/add.png')
    btn_create = Button(root, image=click_btn, borderwidth=0)
    btn_create.place(x=730, y=280)






    btn_evalutate = tk.ttk.Button(root, text="Tester l'accès", width=34)
    btn_evalutate.place(x=730, y=400, height=35)

    # --------------- Work in progress -----------------

    btn_switch = tk.ttk.Button(root, text='Switch to Visage', width=34)

    btn_switch.place(x=730, y=340, height=35)

    btn_option = tk.ttk.Button(root, text="Options", width=15, command = lambda: NewWindow(root))
    btn_option.place(x=730, y=150, height=30)

    # --------------- End work in progress -------------------

    statusbar = tk.Label(root, text="Welcome to FPMs Edge IA Video Surveillance System", relief='sunken', anchor='w',

                         font='Times 10 italic')
    statusbar.pack(side=tk.BOTTOM, fill=tk.X)
    root.mainloop()

    # ►►►► STOP ◄◄◄◄

    cap.release()


if __name__ == '__main__':
    mainWindow()



# UI COLORS
# 08415C  bleu marine cool
# D7D6D6   blanc cassé de bg
# F8E9E9  blanc/rose stylé
# 19647E  bleu zarb
# bleu très foncé