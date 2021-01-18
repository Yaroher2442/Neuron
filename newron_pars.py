import numpy as np
import threading
import queue

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
# from numba import jit

# pre-trained neuron
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

# Задаем seed для повторяемости результатов
np.random.seed(42)


class Neuron_Countours(threading.Thread):
    def __init__(self, queue, frame_num, model):
        threading.Thread.__init__(self)
        self.q_queue = queue
        self.frame_num = frame_num
        self.model = model
        print("Initialized thread" + str(self))

    def run(self):
        while not self.q_queue.empty():
            try:
                # img = image.load_img(self.q_queue.get_nowait(), target_size=(224, 224))
                # x = image.img_to_array(img)
                # x = np.expand_dims(x, axis=0)
                x = preprocess_input(self.q_queue.get_nowait())
                preds = self.model.predict(x)
                preds = decode_predictions(preds, top=3)[0]
                # print(f'файл {self.file}')
                print('Результаты распознавания:', preds)
                print('-----------------------------------------------')
                return
            except:
                return


class Neuron_fullimage(threading.Thread):
    def __init__(self, queue, model):
        threading.Thread.__init__(self)
        self.q_queue = queue
        self.model = model
        print("Initialized thread" + str(self))

    def run(self):
        while not self.q_queue.empty():
            try:
                img = image.load_img(self.q_queue.get_nowait(), target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                preds = self.model.predict(x)
                preds = decode_predictions(preds, top=3)[0]
                # print(f'файл {self.file}')
                print('Результаты распознавания:', preds)
                print('-----------------------------------------------')
                return
            except:
                return