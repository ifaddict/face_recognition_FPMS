import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import cv2
from PIL import Image
from PIL import ImageTk
import real_time_face_recognition as rt
import threading, queue
import face
#pip install tk
#pip install python-opencv
#pip install pillow




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

def launchSampler(video_capture, q, entryLabel):
    if entryLabel == "":
        tk.messagebox.showinfo('Message', 'Le label ne peut pas être vide')
    else:
        threading.Thread(target=rt.captureSamples, args=(video_capture, q, entryLabel,)).start()


def launchEvaluator(video_capture, q, face_recognition):
    threading.Thread(target=rt.evaluateAcess, args=(video_capture, q, face_recognition,)).start()


def CamWindow():
    
    root = tk.Tk()
    
    root.wm_iconbitmap('@fpms.xbm')
    root.title("EDGE IA")
    root.geometry('1040x480')

    q = queue.Queue()
    stateText = ""
    q.put(stateText)

    #=======VARIABLES POUR LA DÉTECTION D'OBJETS

    print("la version de cv2 : ", cv2.__version__)

    net = cv2.dnn.readNet(r"C:\Users\ifadd\Desktop\projetIA\face_recognition_FPMS\faceDetection\contributed\yolov4-custom.weights", r"C:\Users\ifadd\Desktop\projetIA\face_recognition_FPMS\faceDetection\contributed\yolov4-custom.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255)


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
                       bg = 'steel blue', fg = 'black', font = ('times',15,'bold'))
    info.place(x= 660, y=30)

    label   = tk.Label(root, text="Label : ", bg = "#7a86ac")
    label.place(x=660, y=200)

    entryLabel = tk.ttk.Entry(root, width=35)
    entryLabel.place(x=730, y=200)

    code = tk.Label(root, text="Code : ", bg="#7a86ac")
    code.place(x=660, y=230)
    entryCode = tk.ttk.Entry(root, width=35)
    entryCode.place(x=730, y=230)

    btn_create = tk.ttk.Button(root, text='Créer un nouveau profil', width=34,
                               command=lambda: launchSampler(cap, q, entryLabel.get()))
    btn_create.place(x=730, y=280, height=35)

    btn_evalutate = tk.ttk.Button(root, text="Tester l'accès", width=34,
                                  command=lambda: launchEvaluator(cap, q, face_recognition))
    btn_evalutate.place(x=730, y=400, height=35)

    statusbar = tk.Label(root, text="Welcome to FPMs Edge IA Video Surveillance System", relief='sunken', anchor='w',
                         font='Times 10 italic')
    statusbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    

    #►►►► START ◄◄◄◄
    face_recognition = face.Recognition()
    #Tracking(root, cap, photo) #On l'update la première fois
    rt.processFrame(root, cap, photo, face_recognition, q, model)
    root.mainloop()
    
    #►►►► STOP ◄◄◄◄
    
    cap.release()
    
if __name__ == "__main__":
    
    CamWindow()
