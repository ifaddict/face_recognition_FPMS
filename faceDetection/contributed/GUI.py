import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image
from PIL import ImageTk

#pip install tk
#pip install python-opencv
#pip install pillow

def MainWindow():
    
    #Création d'une fenêtre avec icône, titre et taille
    
    window = tk.Tk()    
    window.wm_iconbitmap('fpms.ico')
    window.title("EDGE IA")
    window.geometry('1040x480')
    
    #Création d'un canvas pour y mettre une image de background
    
    can = tk.Canvas(window, width = 1040, height = 480, bg = "#377aff")
    bg = tk.PhotoImage(file="bg.gif")
    can.create_image(0,0, anchor = 'nw', image = bg)
    can.place(x=0,y=0)
    
    #Username et Password demandés avec password caché

    labelUsername   = tk.Label(window, text="Username : ", bg = "#377aff", font = ('times',25,'bold'))
    labelUsername.place(x=200,y=100)
    
    entryUsername = tk.ttk.Entry(window, width = 35)
    entryUsername.place(x=400, y=110)
    
    labelPassword   = tk.Label(window, text='Password : ', bg = "#377aff", font = ('times',25,'bold'))
    labelPassword.place(x=200,y=200)
    
    entryPassword =tk.ttk.Entry(window, show="*", width = 35)
    entryPassword.place(x=400, y=210)
    
    #Bouton pour se connecter. Ce bouton appelle la fonction Verification qui refusera l'accès
    #Si l'username ou le mot de passe est incorrect.

    
    login = tk.Button(window, text ="Log in",  
        command = lambda : Verification(entryUsername,window), fg ="white", bg ="blue",  
        width = 34, height = 2, activebackground = "dodger blue",  
        font =('times', 15, ' bold ')) 
    login.place(x = 200, y = 300)
    
    #Start
    
    window.mainloop()
    

def Verification(txt,window):
    
    #Infos bonnes = accès au système
    
    if txt.get() == 'admin' :
        window.destroy()
        CamWindow()
    
    else :
        lblError = tk.Label(window, text = '     Wrong username or password     ', width = 34,
                            height = 2, fg = 'red', bg = 'white', font = ('times',15,'bold'))
        lblError.place(x = 200, y = 400)


def Tracking(root, cap, photo):
    #Fichier xml pour la détection de visages de face
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml") 
    
    ret, frame = cap.read()
    tickmark   = cv2.getTickCount()
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face       = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors = 4)
    
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    fps   = cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, 'FPS: {:05.2f}'.format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    photo.paste(image)
    
    #La fonction est rappelée toutes les 5 millisecondes
    root.after(5, lambda : Tracking(root,cap,photo))
    
def CamWindow():
    
    root = tk.Tk()
    
    root.wm_iconbitmap('fpms.ico')
    root.title("EDGE IA")
    root.geometry('1040x480')
    
    #On capture la vidéo du périphérique 0 (webcam intégrée)
    cap = cv2.VideoCapture(0)

    # On lit la première frame
    ret, frame = cap.read()

    # ►►►► GUI ◄◄◄◄
    
    #On passe par Pillow pour avoir des images sous Tkinter
    image  = Image.fromarray(frame)
    photo  = ImageTk.PhotoImage(image)
    
    #Création d'un canvas pour y afficher le flux vidéo
    canvas = tk.Canvas(root, width=photo.width()-10, height=photo.height()-10, bd = 3, relief = 'ridge')
    canvas.pack(side='left', fill='both', expand=True)
    
    
    canvas.create_image((0,0), image=photo, anchor='nw')
    
    
    #Création d'un canvas pour y afficher le reste de l'interface
    canvas2 = tk.Canvas(root, width = 400, height = 455)
    canvas2.pack(anchor='ne', fill='both', expand=True)
    
    #imgFpms sera le background de l'interface
    imgFpms = ImageTk.PhotoImage(Image.open("fpmsbg.png"))
    
    canvas2.create_image((400,0), image = imgFpms, anchor='ne')
    
    info    = tk.Label(root, text="Visage inconnu ? \n Choisissez un nom et \n placez-vous devant la \n caméra",
                       bg = 'steel blue', fg = 'red', font = ('times',15,'bold'))
    info.place(x= 660, y=30)
    
    label   = tk.Label(root, text="Label : ", bg = "#7a86ac")
    label.place(x=660, y=250)
    
    entryLabel = tk.ttk.Entry(root, width = 35)
    entryLabel.place(x=730, y=250)
    
    btn    = tk.ttk.Button(root, text='Create new profile', width = 34)
    btn.place(x=730, y = 300, height = 55)
    
    statusbar = tk.Label(root, text="Welcome to FPMs Edge IA Video Surveillance System", relief = 'sunken', anchor = 'w', font = 'Times 10 italic')
    statusbar.pack(side=tk.BOTTOM, fill = tk.X)
    
    

    #►►►► START ◄◄◄◄

    Tracking(root, cap, photo) #On l'update la première fois
    root.mainloop()
    
    #►►►► STOP ◄◄◄◄
    
    cap.release()
    
if __name__ == "__main__":
    
    MainWindow()
