import tkinter as tk
from tkinter import *
from tkinter import ttk
import cv2
from PIL import Image
from PIL import ImageTk
import argparse
import os

import tkinter.messagebox

import threading
import GUI


def launchSampler(entryLabel, entryCode):
    if entryCode == "123456":
        global sampling
        if entryLabel == "":
            tk.messagebox.showinfo('Message', 'Le label ne peut pas être vide')
            return
        sampling = True
    else:
        tk.messagebox.showinfo('Message', 'Mauvais code')


def launchEvaluator():
    print("test")
    global evaluating
    evaluating = True


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



class SampleApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.frames = {}
        self.mainWindow()

    def mainWindow(self):
        self.wm_iconbitmap('@fpms.xbm')
        self.title("EDGE IA")
        self.geometry('1500x900')
        stateText = ""
        print('lol')
        self.cap = cv2.VideoCapture(cv2.CAP_ANY)
        print(self.cap.getBackendName())
        # On lit la première frame
        ret, frame = self.cap.read()
        print('lol2')
        self.retourCam = frame
        print('lol3')
        # ►►►► GUI ◄◄◄◄
        header = tk.Frame(self, background="#273ead", bd=0, relief ="sunken", padx=20, pady=20)
        content = tk.Frame(self, bd =0, relief ="sunken")

        button_frame = tk.Frame(content, background="#273ead", bd=0, relief="sunken")
        lobby_frame = tk.Frame(content, background="#ffffff", bd=0, relief="sunken")
        view_frame = tk.Frame(content, background="#273ead", bd=0, relief="sunken")

        self.cap.release()
        header.grid(row=0, column=0, rowspan=1, sticky="nsew", padx=0, pady=0)
        content.grid(row=1, column=0, rowspan=1, sticky='nsew', padx=0, pady=0)


        button_frame.grid(row=0, column=0, rowspan=1, sticky="nsew", padx=0, pady=0)
        lobby_frame.grid(row=0, column=1, rowspan=1, sticky="nsew", padx=0, pady=0)
        view_frame.grid(row=0, column=2, rowspan=1, sticky="nsew", padx=(0,0), pady=0)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=100)
        content.grid_rowconfigure(0, weight=5)

        self.grid_columnconfigure(0, weight=5)
        content.grid_columnconfigure(0, weight=0, minsize=100)
        content.grid_columnconfigure(1, weight=0, minsize=300)
        content.grid_columnconfigure(2, weight=100)

        view_frame.grid_rowconfigure(0, weight=1)
        view_frame.grid_columnconfigure(0, weight=1)



        #------- Header ---------------
        headerContainer = tk.Frame(header, height=80, width=1500, bg="#273ead")
        headerContainer.pack()

        log = Image.open("logo.png")
        photoLogo = ImageTk.PhotoImage(log)
        label = Label(headerContainer, image=photoLogo, bd=0, bg='#273ead', width=80, height=80)
        label.image = photoLogo
        label.place(x=-10, y=0)

        title = tk.Label(headerContainer, font='Helvetica 18 bold', text = "Système de Surveillance Edge IA",bg=header['bg'], fg='white', pady=5)
        title.place(x=600, y=10)



        #--------- End Header ------------



        #--------- Content ------------



        #------- Menu Frame -----------
        gun_image = PhotoImage(file='icons/gun.png')
        btn_gun = Button(button_frame, image=gun_image, borderwidth=0, cursor="hand2", command=lambda: self.show_frame("ObjectView"))
        btn_gun.place(x=0, y=0, anchor=NW, width=100, height=100)

        face_image = PhotoImage(file='icons/face-detection.png')
        btn_face = Button(button_frame, image=face_image, borderwidth=0, cursor="hand2", command=lambda: self.show_frame("FaceView"))
        btn_face.place(x=0, y=100, anchor=NW, width=100, height=100)

        lock_image = PhotoImage(file='icons/lock.png')
        btn_lock = Button(button_frame, image=lock_image, borderwidth=0, cursor="hand2", command=lambda: self.show_frame("LockView"))
        btn_lock.place(x=0, y=200, anchor=NW, width=100, height=100)

        add_image = PhotoImage(file='icons/add.png')
        btn_add = Button(button_frame, image=add_image, borderwidth=0, cursor="hand2", command=lambda: self.show_frame("AddView"))
        btn_add.place(x=0, y=300, anchor=NW, width=100, height=100)

        settings_image = PhotoImage(file='icons/settings.png')
        btn_settings = Button(button_frame, image=settings_image, borderwidth=0, cursor="hand2", command=lambda: self.show_frame("SettingsView"))
        btn_settings.place(x=0, y=400, anchor=NW, width=100, height=100)

        help_image = PhotoImage(file='icons/aide.png')
        btn_help = Button(button_frame, image=help_image, borderwidth=0, cursor="hand2", command=lambda: self.show_frame("HelpView"))
        btn_help.place(x=0, y=500, anchor=NW, width=100, height=100)

        #       --------------- View Frame -------------------

        #On passe par Pillow pour avoir des images sous Tkinter
        img = Image.fromarray(self.retourCam)
        self.photo = ImageTk.PhotoImage(img)

        canvas = tk.Canvas(view_frame, bd = 0,bg="#273ead", relief = 'ridge')
        canvas.pack(fill='both', expand=True, side="right")
        canvas.create_image((830,150), image=self.photo, anchor='ne')

        # ►►►► START ◄◄◄◄
        self.model, self.device = GUI.initialiseYoloV5()


        visageThread = threading.Thread(target = GUI.processFrameV2, args=(self.photo,))
        visageThread.start()

        # ►►►► STOP ◄◄◄◄





        for F in (ObjectView, FaceView, LockView, AddView, SettingsView, HelpView):
            page_name = F.__name__
            frame = F(main= self, parent=lobby_frame, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("FaceView")

        self.mainloop()
        # ►►►► STOP ◄◄◄◄
        self.cap.release()

    def show_frame(self, page_name):
        if page_name == 'AddView' or page_name == 'SettingsView':
            if GUI.admin is not True:
                tk.messagebox.showwarning("Opération impossible !", "Vous n'êtes pas identifié, vous ne pouvez donc pas accéder à cet onglet.")
                return
            
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.update()
        frame.tkraise()

class ObjectView(tk.Frame):
    def __init__(self,main, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.font= "Helvetica 14 bold"
        self.color = "#273ead"

        self.grid_columnconfigure(0, weight=100)
        self.grid_rowconfigure(0, weight=1, minsize=100)
        self.grid_rowconfigure(1, weight=20)
        self.grid_rowconfigure(2, weight=70)

        canvas_title = tk.Canvas(self,width=300, height=50, bg='white')  
        canvas_title.pack(side='top', fill='both', expand=True)
        canvas_title.create_text(185, 25, text="Detection d'objets dangereux", font='Helvetica 14 bold', fill =self.color)
        controller.update()

        canvas_title = tk.Canvas(self, width=300, height=200,
                                 bg='white')  # le canvas, il faut régler sa taille pour qu'il occupe toute la fenêtre
        canvas_title.pack(fill='both', expand=True)

        canvas_title.create_text(35, 40, text="Etat", font='Helvetica 14 bold', fill ='#000000')
        btn_gun = Button(self, borderwidth=0, cursor="hand2", text="Activer", bg="#273ead", fg="white",
                         font="Helvetica 13 bold", command=lambda: GUI.launchObjectDetect(controller.photo, controller.model, controller.device, main.cap))
        btn_gun.place(x=185, y=200, anchor='center', width=120, height=50)

        if GUI._State:
            etat = 'Activé'
        else:
            etat = 'Désactivé'

        state = canvas_title.create_text(190, 80, anchor='center', width=300,font=self.font, text=etat, justify='center')
        canvas_console = tk.Canvas(self, width=300, height=750,
                                 bg='white')  # le canvas, il faut régler sa taille pour qu'il occupe toute la fenêtre



        canvas_console.pack(fill='both', expand=True)

        seuilObjet = str(GUI.objectThreshold) #opt.confthres

        canvas_console.create_text(105, 40, text="Seuil de confiance" ,font='Helvetica 14 bold', fill ='#000000')

        seuil = canvas_console.create_text(185, 80, text=seuilObjet, font=self.font,fill = '#000000')

        canvas_console.create_text(117, 140, text="Objets pris en charge", font=self.font, fill= "#000000")
        with open("label.txt", 'r') as f:
            labels = [x.rstrip('\n') for x in f.readlines()]
        listBox = Listbox(self)
        for i, label in enumerate(labels):
            listBox.insert(i, label)
        objetsWindow = canvas_console.create_window(185, 250, window= listBox)

        main.after(5, lambda: self.updateView(state, canvas_title, main, canvas_console, seuil))
    def updateView(self, state, canvas, main, canvas_console, seuil):
        if GUI._State:
            canvas.itemconfigure(state, text='Activé')

        else:
            canvas.itemconfigure(state, text='Désactivé')

        canvas_console.itemconfigure(seuil, text=str(GUI.objectThreshold))
        main.after(5, lambda: self.updateView(state, canvas, main, canvas_console, seuil))


class FaceView(tk.Frame):
    def __init__(self,main, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.font= "Helvetica 14 bold"
        self.color = "#273ead"

        self.grid_columnconfigure(0, weight=100)
        self.grid_rowconfigure(0, weight=1, minsize=100)
        self.grid_rowconfigure(1, weight=20)
        self.grid_rowconfigure(2, weight=70)

        canvas_title = tk.Canvas(self,width=300, height=50, bg='white')  # le canvas, il faut régler sa taille pour qu'il occupe toute la fenêtre
        canvas_title.pack(side='top', fill='both', expand=True)
        canvas_title.create_text(185, 25, text="Detection de visages", font='Helvetica 14 bold', fill =self.color)

        canvas_title = tk.Canvas(self, width=300, height=200,
                                 bg='white')  # le canvas, il faut régler sa taille pour qu'il occupe toute la fenêtre
        canvas_title.pack(fill='both', expand=True)

        if GUI._State:
            etat = 'Désactivé'
        else:
            etat = 'Activé'
        canvas_title.create_text(35, 40, text="Etat", font='Helvetica 14 bold', fill ='#000000')
        btn_gun = Button(self, borderwidth=0, cursor="hand2", text="Activer", bg="#273ead", fg="white",
                         font="Helvetica 13 bold", command=lambda: GUI.launchFaceDetect(controller.photo, main.cap))
        btn_gun.place(x=185, y=200, anchor='center', width=120, height=50)

        state = canvas_title.create_text(190, 80, anchor='center', width=300,font=self.font, text=etat, justify='center')
        canvas_console = tk.Canvas(self, width=300, height=750,
                                 bg='white')  # le canvas, il faut régler sa taille pour qu'il occupe toute la fenêtre



        canvas_console.pack(fill='both', expand=True)

        seuilFace = str(GUI.faceThreshold) #opt.confthres

        canvas_console.create_text(105, 40, text="Seuil de confiance" ,font='Helvetica 14 bold', fill ='#000000')

        seuil = canvas_console.create_text(185, 80, text=seuilFace, font=self.font,fill = '#000000')

        canvas_console.create_text(135, 140, text="Membres pris en charge", font=self.font, fill= "#000000")
        '''with open("label.txt", 'r') as f:
            labels = [x.rstrip('\n') for x in f.readlines()]'''
        labels = os.listdir("../PERSONS_ALIGNED/")
        print(labels)
        listBox = Listbox(self)
        for i, label in enumerate(labels):
            listBox.insert(i, label)
        objetsWindow = canvas_console.create_window(185, 250, window= listBox)

        main.after(5, lambda: self.updateView(state, canvas_title, main, canvas_console, seuil))

    def updateView(self, state, canvas, main, canvas_console, seuil):
        if GUI._State:
            canvas.itemconfigure(state, text='Désactivé')

        else:
            canvas.itemconfigure(state, text='Activé')
        canvas_console.itemconfigure(seuil, text=str(GUI.faceThreshold))

        main.after(5, lambda: self.updateView(state, canvas, main, canvas_console, seuil))






class LockView(tk.Frame):
    def __init__(self, main, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.font = "Helvetica 14 bold"
        self.color = "#273ead"

        self.grid_columnconfigure(0, weight=100)
        self.grid_rowconfigure(0, weight=1, minsize=100)
        self.grid_rowconfigure(1, weight=20)
        self.grid_rowconfigure(2, weight=70)

        canvas_title = tk.Canvas(self, width=300, height=20,
                                 bg='white')  # le canvas, il faut régler sa taille pour qu'il occupe toute la fenêtre
        canvas_title.pack(side='top', fill='both', expand=True)
        canvas_title.create_text(185, 25, text="Évaluation", font='Helvetica 14 bold', fill=self.color)

        canvas_console = tk.Canvas(self, width=300, height=750,
                                   bg='white')  # le canvas, il faut régler sa taille pour qu'il occupe toute la fenêtre

        canvas_console.pack(fill='both', expand=True)

        canvas_console.create_text(185, 365, text="Logs", font=self.font, fill="#000000")

        log_canvas = tk.Canvas(self, width=280, height=25)
        log = log_canvas.create_text(140, 12, anchor='center', state='disabled', width=280, text="log",
                                     justify='center')
        Console = canvas_console.create_window(185, 425, window=log_canvas)


        btn_gun = Button(self, borderwidth=0, cursor="hand2", text="Évaluer", bg="#273ead", fg="white", font="Helvetica 13 bold", command=lambda: GUI.launchEvaluator())
        btn_gun.place(x=130, y=250, anchor=NW, width=120, height=50)
        
        btn_logout =  Button(self, borderwidth=0, cursor="hand2", text="Déconnexion", bg="#273ead", fg="white", font="Helvetica 13 bold", command=lambda: self.deco())
        btn_logout.place(x=130, y=400, anchor=NW, width=120, height=50)
        main.after(5, lambda: self.updateView(log_canvas, log, main, canvas_console))
        



    def updateView(self, log_canvas, log, main, canvas_console):

        log_canvas.itemconfigure(log, text=str(GUI.faceLogs))

        main.after(5, lambda: self.updateView(log_canvas, log, main, canvas_console))
        
    def deco(self):
        GUI.admin = False
        GUI.faceLogs = 'Déconnecté'
        

class AddView(tk.Frame):

    def __init__(self,main, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        canvasTitle = tk.Canvas(self, bg="#ffffff", height=50)  # le canvas, il faut régler sa taille pour qu'il occupe toute la fenêtre
        canvasTitle.create_text(185, 25, text="Gérer les membres", font="Helvetica 15 bold", fill="#273ead")  # on ajoute le texte



        self.grid_columnconfigure(0, weight=0, minsize=50)
        self.grid_rowconfigure(1, weight=0, minsize=540)
        self.grid_rowconfigure(2, weight=0, minsize=150)
        self.grid_rowconfigure(3, weight=0, minsize=10)
        canvasTitle.grid(row=0, column=0, rowspan=1, sticky="nsew", padx=0, pady=0)

        addFrame = tk.Frame(self, bg="#ffffff", height=300)

        labelAjout = tk.Label(addFrame, text="Ajouter un membre ", bg="#ffffff", fg="black", font="Helvetica 13 bold")
        labelAjout.place(x=190, y=50, anchor='center')

        labelName = tk.Label(addFrame, text="Label : ", bg="#ffffff", fg="black", font="Helvetica 13 bold")
        labelName.place(x=30, y=87)
        entryLabel = tk.ttk.Entry(addFrame, width=30)
        entryLabel.place(x=120, y=90)

        code = tk.Label(addFrame, text="Code : ", bg="#ffffff", fg="black", font="Helvetica 13 bold")
        code.place(x=30, y=127)
        entryCode = tk.ttk.Entry(addFrame, width=30, show="*")
        entryCode.place(x=120, y=130)


        btn_add = Button(addFrame, borderwidth=0, cursor="hand2", text="Ajouter", bg="#273ead", fg="white", font="Helvetica 13 bold", command=lambda: add(entryLabel, entryCode.get()))
        btn_add.place(x=190, y=225, anchor='center', width=120, height=50)


        labelDelete = tk.Label(addFrame, text="Supprimer un membre ", bg="#ffffff", fg="black", font="Helvetica 13 bold")
        labelDelete.place(x=190, y=350, anchor='center')

        label_menu = ttk.Combobox(self, state='readonly', postcommand=lambda :refresh(label_menu))
        label_menu.place(x=190, y=450, height=25, width=180, anchor='center')
        liste_label = os.listdir(r'../PERSONS_ALIGNED')
        label_Variable = tk.StringVar(addFrame)

        if len(liste_label)==0:
            label_Variable.set('Aucun visage enregistré.')
            label_menu.set(label_Variable.get())

        else:
            label_menu['values'] = liste_label
            label_menu.current(0)




        btn_supp = Button(addFrame, borderwidth=0, cursor="hand2", text="Supprimer", bg="#273ead", fg="white", font="Helvetica 13 bold", command=lambda: supp(label_menu))
        btn_supp.place(x=190, y=475, anchor='center', width=120, height=50)


        if label_menu.get() == 'Aucun visage enregistré.':
            btn_supp['state'] = tk.DISABLED
        addFrame.grid(row=1, column=0, rowspan=1, sticky="nsew", padx=0, pady=0)

        canvas_console = tk.Canvas(self, width=280, height=25, bg='white')  # le canvas, il faut régler sa taille pour qu'il occupe toute la fenêtre
        canvas_console.grid(row=2, column=0, rowspan=1, sticky="nsew", padx=0, pady=0)

        canvas_console.create_text(175, 20, text="Logs", font="Helvetica 14 bold", fill="#000000")
        log = canvas_console.create_text(175, 75, anchor='center', state='disabled', width=280, text="log",
                                     justify='center')
        main.after(5, lambda: self.updateView(canvas_console, log, main, canvas_console))


    def updateView(self, log_canvas, log, main, canvas_console):

        log_canvas.itemconfigure(log, text=str(GUI.faceLogs))

        main.after(5, lambda: self.updateView(log_canvas, log, main, canvas_console))





def refresh(label_menu):
    label_menu['values'] = os.listdir(r'../PERSONS_ALIGNED')
def add(entryLabel, code):
    if entryLabel.get() in os.listdir(r'../PERSONS_ALIGNED'):
        tk.messagebox.showwarning("Opération impossible !",
                                  "Le label existe déjà dans la base de données.")
        return

    if GUI._State is not True:
        GUI.launchSampler(entryLabel, code)
    else:
        tk.messagebox.showwarning("Opération impossible !", "La détection de visage doit être lancée pour utiliser cette fonction. Veuillez la lancer depuis l'onglet correspondant.")

def supp(label_menu):
    label = label_menu.get()

    if label != 'Aucun visage enregistré.':
        if GUI._State is not True: #Ne pas lancer la suppression si la detection n'est pas lancée
            classes = os.listdir(r'../PERSONS_ALIGNED')
            if len(classes)  > 2:
                label_menu['values'] = os.listdir(r'../PERSONS_ALIGNED')
                GUI.suppress(label)
            else:
                tk.messagebox.showwarning("Opération impossible !", "Le ré-entrainement n'autorise pas un nombre de classe inférieur ou égal à 1. Veuillez enregistrer une personne supplémentaire avant d'en supprimer une.")

        else:
            tk.messagebox.showwarning("Opération impossible !", "La détection de visage doit être lancée pour utiliser cette fonction. Veuillez la lancer depuis l'onglet correspondant.")



class SettingsView(tk.Frame):

    def __init__(self,main, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        canvasTitle = tk.Canvas(self, bg="#ffffff", height=50)  # le canvas, il faut régler sa taille pour qu'il occupe toute la fenêtre
        canvasTitle.create_text(185, 25, text="Options", font="Helvetica 15 bold", fill="#273ead")  # on ajoute le texte

        self.grid_columnconfigure(0, weight=0, minsize=50)
        self.grid_rowconfigure(1, weight=5)
        self.grid_rowconfigure(2, weight=5)
        canvasTitle.grid(row=0, column=0, rowspan=1, sticky="nsew", padx=0, pady=0)

        optionsFrame = tk.Frame(self, bg="#ffffff", height=400)

        objectsThreshHolds = ["0.2", "0.3", "0.4", "0.5", "0.6"]
        objetCombo = ttk.Combobox(optionsFrame, values=objectsThreshHolds)
        currentConf = str(GUI.objectThreshold)
        objetCombo.current(objectsThreshHolds.index(currentConf))
        objetCombo.place(x=160, y=90)

        objetLabel = tk.Label(optionsFrame, text="Seuil de confiance \nDétection d'objets", bg="#ffffff")
        objetLabel.place(x=30, y=80)


        threshHolds = ["0.5", "0.7", "0.75", "0.8", "0.85", "0.9"]
        visageCombo = ttk.Combobox(optionsFrame, values=threshHolds)
        currentConf = str(GUI.faceThreshold)
        visageCombo.current(threshHolds.index(currentConf))
        visageCombo.place(x=160, y=150)

        visageLabel = tk.Label(optionsFrame, text="Seuil de confiance \nDétection visages", bg="#ffffff")
        visageLabel.place(x=30, y=140)


        framesScale = Scale(optionsFrame, orient='horizontal', from_=30, to=100,
              resolution=1, tickinterval=20, length=250,
              label='Nombre de frames pour le ré-entrainement', background='white', bd=0)

        framesScale.place(x=55, y=230)
        framesScale.set(GUI.framesOption)



        btn_gun = Button(optionsFrame, borderwidth=0, cursor="hand2", text="Sauvegarder", bg="#273ead", fg="white", font="Helvetica 13 bold", command =lambda : GUI.saveAndClose(visageCombo.get(), objetCombo.get(),framesScale.get(), True))
        btn_gun.place(x=120, y=350, anchor=NW, width=120, height=50)


        optionsFrame.grid(row=1, column=0, rowspan=1, sticky="nsew", padx=0, pady=0)





class HelpView(tk.Frame):
    def __init__(self,main, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        canvasTitle = tk.Canvas(self, bg="#ffffff", height=50)  # le canvas, il faut régler sa taille pour qu'il occupe toute la fenêtre
        canvasTitle.create_text(185, 25, text="Aide", font="Helvetica 15 bold", fill="#273ead")  # on ajoute le texte

        self.grid_columnconfigure(0, weight=0, minsize=50)
        self.grid_rowconfigure(1, weight=5)
        self.grid_rowconfigure(2, weight=5)

        canvasTitle.grid(row=0, column=0, rowspan=1, sticky="nsew", padx=0, pady=0)
        canvasMain = tk.Canvas(self, bg="#ffffff", height=395)
        canvasMain.create_text(40, 140,
                               text="Le programme de vidéo-surveillance \nEdge IA est "
                                    "disponible dans 2 modes \nun mode 'Détection d'objets' et un \nmode 'Reconnaissance faciale'"
                                    "les\n options de test d'accès et d'ajout \nde membres sont uniquement disponibles \nen mode "
                                    "'Reconnaissance faciale' \n \n"
                                    "Une section 'Options' est disponible \npour que l'administrateur puisse  \nmodifier les paramètres"
                                    "de détection.",
                               font="Helvetica 12",
                               anchor=W)
        canvasMain.grid(row=1, column=0, rowspan=1, sticky="nsew", padx=0, pady=0)



def getOptions():
    options = []
    with open('config.conf', 'r') as config:
        for line in config:
            options.append(float(line.split("=")[1].strip()))
    return options

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='testYaya.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')

    #►►►► Work in Progress : 3) Bouton Option to alter Thresholds ◄◄◄◄

    options = getOptions()
    faceThreshold = options[0]
    objectThreshold = options[1]
    framesOption = options[2]

    #►►►► End Work in Progress : 3) Bouton Option to alter Thresholds ◄◄◄◄

    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()

    _State = True #Use to identify if object or faces are processed.
    sampling = False #Use to alert that a retraining of the visage model is requested.
    retrained = False #Use to reboot the model when retrained
    _Option = False #Use to specify that some options changed
    evaluating = False #Used to check if we need to evaluate the access


    # face_recognition = face.Recognition(GUI.faceThreshold)
    app = SampleApp()
    app.mainloop()

# UI COLORS
# 08415C  bleu marine cool
# D7D6D6   blanc cassé de bg
# F8E9E9  blanc/rose stylé
# 19647E  bleu zarb
# bleu très foncé
