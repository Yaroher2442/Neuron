import numpy as np
import os
import cv2
import queue
import time

from newron_pars import Neuron_Countours, Neuron_fullimage
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions


# pwd_queue = queue.Queue()
def Pretrained_files():
    pwd_queue = queue.Queue()
    model = VGG16(weights='imagenet')
    path = os.path.join(os.getcwd(), 'testing_pretrained')
    for f_name in os.listdir(path):
        pwd_queue.put_nowait(os.path.join(path, f_name))
        image = cv2.imread(os.path.join(path, f_name))
        worker = Neuron_fullimage(pwd_queue, model)
        worker.setDaemon(True)
        worker.start()
        cv2.imshow('frame', image)
        cv2.waitKey()
        cv2.destroyAllWindows()


def Pretrained_camera_Contours():
    q_queue = queue.Queue()
    model = VGG16(weights='imagenet')
    counter = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        counter += 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        # ls = [cn for cn in [cv2.boundingRect(contour) for contour in contours] if (cn[0] > 50 and cn[1] > 50) and (cn[2] > 50 and cn[3] >50)]
        # print(len(ls))
        output = frame.copy()
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            if (w > 60 and h > 60) and (x != 0 and y != 0):
                file_name = f'chop_{counter}_{idx}.jpg'
                cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 0, 0), 1)
                cv2.imshow('chop', output[y:y + h, x:x + w])
                cv2.imwrite(os.path.join(os.getcwd(), 'buffers', file_name), output[y:y + h, x:x + w])
                img = image.load_img(os.path.join(os.getcwd(), 'buffers', file_name), target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                q_queue.put(x)
                worker = Neuron_Countours(q_queue, counter, model)
                worker.setDaemon(True)
                worker.start()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def Pretrained_camera_con():
    q_queue = queue.Queue()
    model = VGG16(weights='imagenet')
    counter = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        counter += 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        # ls = [cn for cn in [cv2.boundingRect(contour) for contour in contours] if (cn[0] > 50 and cn[1] > 50) and (cn[2] > 50 and cn[3] >50)]
        # print(len(ls))
        output = frame.copy()
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            if (w > 60 and h > 60) and (x != 0 and y != 0):
                file_name = f'chop_{counter}_{idx}.jpg'
                cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 0, 0), 1)
                cv2.imshow('chop', output[y:y + h, x:x + w])
                cv2.imwrite(os.path.join(os.getcwd(), 'buffers', file_name), output[y:y + h, x:x + w])
                img = image.load_img(os.path.join(os.getcwd(), 'buffers', file_name), target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                preds = model.predict(x)
                preds = decode_predictions(preds, top=3)[0]
                print('Результаты распознавания:', preds)
                print('-----------------------------------------------')
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def Pretrained_camera():
    q_queue = queue.Queue()
    model = VGG16(weights='imagenet')
    counter = 0
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        f_name = os.path.join(os.getcwd(), 'buffers', f'{counter}.jpg')
        cv2.imwrite(f_name, frame)
        q_queue.put_nowait(f_name)
        worker = Neuron_fullimage(q_queue, model)
        worker.setDaemon(True)
        worker.start()
        cv2.imshow('frame', frame)
        time.sleep(0.2)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
