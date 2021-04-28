import argparse
import os
import random
import time
import configargparse
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox

import cv2
from PIL import Image
from PIL import ImageTk
import numpy as np
import real_time_face_recognition as rt
import threading, queue
import face
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


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
        objetCONF_Variable.set(str(opt.conf_thres) + " (actuel)")
        objetCONF_Option = tk.OptionMenu(self,
                                      objetCONF_Variable,
                                      "0.2", "0.3", "0.4 (recommandé)", "0.5")
        objetCONF_Option.pack()

        # Creating a Option Menu for visageCONF
        # Set the variable for visageCONF and create the list
        # of options by initializing the constructor 
        # of class OptionMenu.
        visageCONF_Variable = tk.StringVar(self)
        visageCONF_Variable.set(str(opt.conf_thres2) + " (actuel)")
        visageCONF_Option = tk.OptionMenu(self, visageCONF_Variable,
                                     "0.5", "0.6", "0.7 (recommandé)",
                                     "0.8")
        visageCONF_Option.pack()

        liste_label = os.listdir(r'../PERSONS_ALIGNED')

        label_Variable = tk.StringVar(self)
        label_Variable.set(liste_label[0])

        label_menu = tk.OptionMenu(self, label_Variable, *liste_label)
        label_menu.place(x=50, y=100, height=35)


        btn_label = tk.ttk.Button(self, text='Supprimer le label', width=17, command = lambda :suppress(label_Variable.get(),self, master))
        btn_label.place(x=150, y=100, height=35)

        btn_save = tk.ttk.Button(self, text='Enregistrer', width=34,
                                   command=lambda: saveAndClose(self, float(visageCONF_Variable.get().split()[0]), float(objetCONF_Variable.get().split()[0]), True))
        btn_save.place(x=50, y=200, height=35)

        btn_quit = tk.ttk.Button(self, text='Quitter', width=34,
                                   command=lambda: saveAndClose(self, float(visageCONF_Variable.get().split()[0]), float(objetCONF_Variable.get().split()[0])))
        btn_quit.place(x = 50, y = 150, height = 35)

def initialiseYoloV5():
    source, weights, view_img, imgsz = opt.source, opt.weights, opt.view_img, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))  # True when real-time.

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    if half:
        model.half()  # to FP16

    # Set Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once




    return model, device

def saveAndClose(window, visageConf, objetConf, save=False):

    global _Option
    if save:
        opt.conf_thres = objetConf
        opt.conf_thres2 = visageConf
        update()
        _Option = True
    window.destroy()

def update():
    with open('config.conf', 'w') as config:
        config.write("conf_thres=" + str(opt.conf_thres) + '\n')
        config.write("conf_thres2=" + str(opt.conf_thres2))

def suppress(label, opt, master):
    for file in os.listdir(r'../PERSONS_ALIGNED/' + label):
        os.remove(r'../PERSONS_ALIGNED/' + label + '/' + file)
    os.rmdir(r'../PERSONS_ALIGNED/' + label)

    threading.Thread(target = retrain, args=()).start()
    opt.destroy()
    NewWindow(master)

def objectDetect(photo, model, device):

    global _Option

    time.sleep(0.2)
    source, weights, view_img, imgsz = opt.source, opt.weights, opt.view_img, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))  # True when real-time.
    # ------------- Begin work in progress --------------

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    half = device.type != 'cpu'  # half precision only supported on CUDA
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    

    global _State
    t0 = time.time()
    # ------------- End work in progess -----------------
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    for path, img, im0s, vid_cap in dataset: #Webcam Stream
        if _State is not True: #Process frame only in asked to do so
            vid_cap.release()
            break

        if _Option is True:
            print("Parameters altered. Resetting model...")
            _Option = False
            vid_cap.release()
            objetThread = threading.Thread(target=objectDetect, args=(photo, model, device))
            objetThread.start()
            return

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:

            img = img.unsqueeze(0)
    
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
    
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
    
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
    

            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    
                # Write results
                for *xyxy, conf, cls in reversed(det): # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
    
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
    
            # Stream results
            frame = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo.paste(image)
            cv2.waitKey(1)  # 1 millisecond

    print(f'Object detection aborted. ({time.time() - t0:.3f}s)')


def processFrameV2(photo, entryLabel):
    global retrained
    global sampling
    global evaluating
    global _Option
    time.sleep(0.2)
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    stride = 32
    source, imgsz = opt.source, opt.img_size
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    t0 = time.time()
    numero = 0
    global face_recognition
    #face_recognition.encoder = face.Encoder()
    count = 0 # utilisé pour l'évaluation d'accès
    for path, img, im0s, vid_cap in dataset: #Webcam Stream
        t2 = time.time()
        if _State is True:
            vid_cap.release()
            return
        if _Option is True:
            print("Parameters altered. Resetting model...")
            _Option = False
            face_recognition = face.Recognition(opt.conf_thres2)
        img = im0s[0].copy()
        faces = face_recognition.identify(img)
        if evaluating:
            if len(faces) == 1:
                if faces[0].name is not None:
                    if faces[0].name != "Inconnu":
                        # changeText(frameQueue, "Visage connu detecte")
                        print("visage connu détecté")
                        count += 1
                    else:
                        # changeText(frameQueue, "Visage Inconnu, accès refusé")
                        print("visage inconnu accès refusé")
                        evaluating = False
                        count = 0
            else:
                # changeText(frameQueue, "Une seule personne autorisee")
                print("Une seule personne autorisée pour l'évaluation d'accès")
                count = 0
            if count == 10:
                print("accès autorisé")
                count = 0
                evaluating = False

        if sampling: #Altered in launchSampler()
            print("Sampling...")
            if len(faces) == 1 and faces[0].name == "Inconnu":
                y = faces[0].bounding_box[0]
                x = faces[0].bounding_box[1]
                h = faces[0].bounding_box[2]
                w = faces[0].bounding_box[3]
                samplesPath = "../PERSONS_ALIGNED/" + entryLabel.get() + "/"
                if not os.path.exists(samplesPath):
                    os.mkdir(samplesPath)
                image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                cropped = rt.resize(image[x:w, y:h, :], (182, 182))
                im = Image.fromarray((cropped * 255).astype(np.uint8))
                im.save(samplesPath + str(numero) + ".png")
                numero += 1
                print("Saved")
                if numero == 30:
                    print("Sampling done. Training...")
                    sampling = False
                    threading.Thread(target=retrain, args=()).start()
        rt.add_overlays(img, faces, frame_rate)
        # on convertit la frame en image PIL et on la paste sur l'interface
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        photo.paste(image)
        t3 = time.time()
        #print(f'{source}: Done. ({t3 - t2:.3f}s)')
        if retrained:
            print("Resetting model...")
            face_recognition = face.Recognition(opt.conf_thres2)
            retrained = False

def retrain():
    global retrained
    rt.retrain("../model_checkpoints/my_classifier.pkl", "../PERSONS_ALIGNED", "../model_checkpoints/20180408-102900.pb")
    retrained = True


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

    global sampling
    if entryLabel == "":
        tk.messagebox.showinfo('Message', 'Le label ne peut pas être vide')
        return
    sampling = True


def launchEvaluator(video_capture, q, face_recognition):
    global evaluating
    evaluating = True

def switch(cap,photo,model, device, q, entryLabel):

    #--------------- Work in progess ---------------
    
    global _State
    #_State is True by default

    if _Option is not True:
        _State = not _State #Switch the variable

    if _State is not True:
        visageThread = threading.Thread(target=processFrameV2, args=(photo,entryLabel))
        visageThread.start()
    
    else:
        objetThread = threading.Thread(target = objectDetect, args = (photo,model, device))
        objetThread.start()



def CamWindow():

    root = tk.Tk()
    
    root.wm_iconbitmap('@fpms.xbm')
    root.title("EDGE IA")
    root.geometry('1040x480')

    q = queue.Queue()
    stateText = ""
    q.put(stateText)

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

    #--------------- Work in progress -----------------

    btn_switch = tk.ttk.Button(root, text='Switch Mode', width=34,
                               command=lambda: switch(cap, photo,model, device, q, entryLabel))

    btn_switch.place(x=730, y=340, height=35)


    btn_option = tk.ttk.Button(root, text="Options", width = 15, command = lambda: NewWindow(root))
    btn_option.place(x=730, y=150, height= 30)

    #--------------- End work in progress -------------------



    statusbar = tk.Label(root, text="Welcome to FPMs Edge IA Video Surveillance System", relief='sunken', anchor='w',

                         font='Times 10 italic')
    statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    

    #►►►► START ◄◄◄◄
    model, device = initialiseYoloV5()

    
    objetThread = threading.Thread(target = objectDetect, args = (photo,model, device))
    objetThread.start()
    root.mainloop()
    
    #►►►► STOP ◄◄◄◄
    
    cap.release()
    
if __name__ == "__main__":

    parser = configargparse.ArgParser(default_config_files=['config.conf'])
    parser.add_argument('--weights', nargs='+', type=str, default='testYaya.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--conf_thres2', type=float, default=0.8, help='visage confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view_img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', default = None, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')


    opt = parser.parse_args()
    print(vars(opt))
    _State = True #Use to identify if object or faces are processed.
    sampling = False #Use to alert that a retraining of the visage model is requested.
    retrained = False #Use to reboot the model when retrained
    _Option = False #Use to specify that some options changed
    evaluating = False #Used to check if we need to evaluate the access

    face_recognition = face.Recognition(opt.conf_thres2)

    CamWindow()
