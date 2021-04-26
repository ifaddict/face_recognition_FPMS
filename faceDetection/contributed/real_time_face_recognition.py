# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import time
import os
import cv2
import sys


sys.path.append("../src")
sys.path.append("..")

from skimage.transform import resize
import face
from PIL import Image
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from src.classifier import main as trainer
import facenet
from src.classifier import split_dataset
import pickle
from sklearn.svm import SVC
import math
import time
import threading, queue


def changeText(q, text):
    temp_text = q.get()
    q.put(text)

def retrain(classifier_filename, data_dir, model, image_size=160, seed=666, min_nrof_images_per_class=20, nrof_train_images_per_class=10, batch_size=90):
    with tf.compat.v1.Graph().as_default():

        with tf.compat.v1.Session() as sess:

            np.random.seed(seed=seed)

            dataset = facenet.get_dataset(data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(classifier_filename)
            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)

            # Create a list of class names
            class_names = [cls.name.replace('_', ' ') for cls in dataset]

            # Saving classifier model
            with open(classifier_filename, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
            reloaded = True


def captureSamples(capture, frameQueue, name):
    NUMBER_OF_SAMPLES = 30
    numero=1
    frame_no = 0
    detection = face.Detection()
    changeText(frameQueue, "Echantillonage, veuillez mouvoir votre visage")
    while capture.isOpened():
        ret, image = capture.read()
        frame_no += 1
        if frame_no % 10 == 0:
            faces = detection.find_faces(image)
            print(faces)
            if len(faces) == 1:
                y = faces[0].bounding_box[0]
                x = faces[0].bounding_box[1]
                h = faces[0].bounding_box[2]
                w = faces[0].bounding_box[3]
                samplesPath = "../PERSONS_ALIGNED/" + name + "/"
                if not os.path.exists(samplesPath):
                    os.mkdir(samplesPath)
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                cropped = resize(image[x:w, y:h, :], (182, 182))
                im = Image.fromarray((cropped * 255).astype(np.uint8))
                im.save(samplesPath + str(numero) + ".png")
                numero += 1

                if numero == NUMBER_OF_SAMPLES:
                    break
                if cv2.waitKey(1) == ord(' '):
                    break

    changeText(frameQueue, "Echantillonage finit")
    time.sleep(2)
    changeText(frameQueue, "Entrainement en cours, veuillez patienter")


    retrain("../model_checkpoints/my_classifier.pkl", "../PERSONS_ALIGNED", "../model_checkpoints/20180408-102900.pb")
    changeText(frameQueue, "Entrainement finit")
    time.sleep(2)
    changeText(frameQueue, "")
    global reloaded
    reloaded = True


def evaluateAcess(capture, frameQueue, face_recognition):
    changeText(frameQueue, "Test de l'acces")
    count = 0
    while capture.isOpened():
        ret, frame = capture.read()
        faces = face_recognition.identify(frame)
        if len(faces) == 1:
            if faces[0].name is not None:
                if faces[0].name != "Inconnu":
                    changeText(frameQueue, "Visage connu detecte")
                    count += 1
                else:
                    changeText(frameQueue, "Visage Inconnu, accès refusé")
                    count = 0
        else:
            changeText(frameQueue, "Une seule personne autorisee")
            count = 0

        if count == 10:
            changeText(frameQueue, "Acces autorise")
            break


def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)


def processFrame(root, video_capture, photo, face_recognition, q):
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    reloaded = False


    start_time = time.time()

    ret, frame = video_capture.read()

    if (frame_count % frame_interval) == 0:
        faces = face_recognition.identify(frame)

        # Check our current fps
        end_time = time.time()
        if (end_time - start_time) > fps_display_interval:
            frame_rate = int(frame_count / (end_time - start_time))
            frame_count = 0

    add_overlays(frame, faces, frame_rate)

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
    root.after(5, lambda : processFrame(root, video_capture, photo, face_recognition, q, model))



if __name__ == '__main__':
    processFrame()
