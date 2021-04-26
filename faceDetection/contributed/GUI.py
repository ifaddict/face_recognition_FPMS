import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import cv2
from PIL import Image
from PIL import ImageTk
import real_time_face_recognition as rt
import threading, queue
import face
import detect as dt
#pip install tk
#pip install python-opencv
#pip install pillow
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import argparse
from pathlib import Path


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
    dt.detect()
    root.mainloop()
    
    #►►►► STOP ◄◄◄◄
    
    cap.release()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='testYaya.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))
    CamWindow()
