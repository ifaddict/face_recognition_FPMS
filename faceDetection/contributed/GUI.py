import argparse
import random
import time

import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import cv2
from PIL import Image
from PIL import ImageTk

import real_time_face_recognition as rt
import threading, queue
import face
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

_State = True

def initialiseYoloV5():
    

    #------------- Begin work in progress --------------

    global _State


    #------------- End work in progess -----------------

    source, weights, view_img, imgsz = opt.source, opt.weights, opt.view_img, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://')) # True when real-time.


    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

def objectDetect(photo):
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    t0 = time.time()


    for path, img, im0s, vid_cap in dataset: #Webcam Stream
        if _State is not True: #Process frame only in asked to do so
            vid_cap.release()
            print(vid_cap.isOpened())
            break



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
            if view_img:
    
                frame = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                photo.paste(image)
                cv2.waitKey(1)  # 1 millisecond

    print(f'Object detection aborted. ({time.time() - t0:.3f}s)')

def processFrame(root, video_capture, photo, face_recognition, q):
    global _State
    
    if _State is True:
        video_capture.release()
        return
    
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    reloaded = False

    
    start_time = time.time()
    
    time.sleep(0.05)
    if video_capture.isOpened() is not True :
        video_capture.open(0)

    ret, frame = video_capture.read()
    
    
    if (frame_count % frame_interval) == 0:
        faces = face_recognition.identify(frame)

        # Check our current fps
        end_time = time.time()
        if (end_time - start_time) > fps_display_interval:
            frame_rate = int(frame_count / (end_time - start_time))
            frame_count = 0

    rt.add_overlays(frame, faces, frame_rate)

    if reloaded == True:
        face_recognition = face.Recognition()
        print("Modèle rechargé")
        reloaded = False
    frame_count += 1
    stateText = q.get()
    q.put(stateText)
    wHeight = video_capture.get(4)
    cv2.putText(frame, stateText, (10, int(wHeight)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                thickness=2, lineType=2)

    # on convertit la frame en image PIL et on la paste sur l'interface
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    photo.paste(image)

    #La fonction est rappelée toutes les 5 millisecondes
    root.after(5, lambda : processFrame(root, video_capture, photo, face_recognition, q))




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

def switch(root,cap,photo,face_recognition,q):
    #--------------- Work in progess ---------------
    
    global _State
    #_State is True by default
    
    _State = not _State #Switch the variable

    if _State is not True:
        cap.release()    
        visageThread = threading.Thread(target=processFrame, args=(root, cap, photo,face_recognition, q))
        visageThread.start()
    
    else:
        cap.release()
        objetThread = threading.Thread(target = objectDetect, args = (photo,))
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

    btn_switch = tk.ttk.Button(root, text='Switch to Visage', width=34,
                               command=lambda: switch(root,cap,photo,face_recognition, q))

    btn_switch.place(x=730, y=340, height=35)

    #--------------- End work in progress -------------------



    statusbar = tk.Label(root, text="Welcome to FPMs Edge IA Video Surveillance System", relief='sunken', anchor='w',
                         font='Times 10 italic')
    statusbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    

    #►►►► START ◄◄◄◄

    face_recognition = face.Recognition()

    #---------------- Work in Progress -----------------
    cap.release()


    #---------------- End work in Progress ------------------
    
    objetThread = threading.Thread(target = objectDetect, args = (photo,))
    objetThread.start()
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
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    CamWindow()
