# импорт необходимых пакетов
from keras.models import load_model
import argparse
import pickle
import cv2
import os
import matplotlib.pyplot as plt

# создание парсера аргументов и их передача
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image we are going to classify")
ap.add_argument("-m", "--model", required=True,
                help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to label binarizer")
ap.add_argument("-w", "--width", type=int, default=28,
                help="target spatial dimension width")
ap.add_argument("-e", "--height", type=int, default=28,
                help="target spatial dimension height")
ap.add_argument("-f", "--flatten", type=int, default=-1,
               help="whether or not we should flatten the image")
args = vars(ap.parse_args())


def process_cells(filename):
    # загрузка входного изображения и изменение его размера на необходимый
    image = cv2.imread(filename)
    output = image.copy()
    image = cv2.resize(image, (args["width"], args["height"]))
    # масштаб значений пикселей к диапазону [0, 1]
    image = image.astype("float") / 255.0
    # проверка необходимости сглаживания изображения и добавление размера
    # пакета
    if args["flatten"] > 0:
        image = image.flatten()
        image = image.reshape((1, image.shape[0]))

    # иначе работа с CNN без сглаживания изображения
    # и добавление размера пакета
    else:
        image = image.reshape((1, image.shape[0], image.shape[1],
                               image.shape[2]))
    # загрузка модели и бинаризатора меток
    print("[INFO] loading network and label binarizer...")
    model = load_model(args["model"])
    lb = pickle.loads(open(args["label_bin"], "rb").read())

    # предсказание изображения
    preds = model.predict(image)
    print(preds)

    # находение индекса меток класса с наибольшей вероятностью
    # соответствия
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]

    # метка класса + вероятность на выходном изображении
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)

    # показ выходного изображения
    cv2.imshow("Image", output)
    cv2.waitKey(0)

path = r'C:\Users\elagina\Documents\keras-tutorial\keras-tutorial\images\\'
for image in os.listdir(path):
    process_cells(path+image)