import os
for dirname, _, filenames in os.walk(r'C:\Users\elagina\Documents\keras-tutorial\keras-tutorial\images\\'):
    for filename in filenames:
        os.path.join(dirname, filename)

import pandas as pd

# импорт бэкенд Agg из matplotlib для сохранения графиков на диск
import matplotlib
matplotlib.use("Agg")

# подключение необходимых пакетов
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

DATADIR = '../input/color-classification/ColorClassification'
CATEGORIES = ['orange','Violet','red','Blue','Green','Black','Brown','White']
IMG_SIZE=100


# рандомное перемешивание изображений
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

path = r'C:\Users\elagina\Documents\keras-tutorial\keras-tutorial\images\\'
IMG_SIZE=100

for label in labels:
    path=os.path.join(data, label)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        plt.imshow(img_array)
        plt.show()
        break
    break

training_data=[]
def create_training_data():
    for label in labels:
        path=os.path.join(data, label)
        class_num=labels.index(label)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()

lenofimage = len(training_data)

data = []
labels = []

for categories, label in training_data:
    data.append(labels)
    labels.append(label)
X= np.array(data).reshape(lenofimage,-1)
X.shape
X = X/255.0

y=np.array(labels)
y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

from sklearn.svm import SVC
svc = SVC(kernel='linear',gamma='auto')
svc.fit(X_train, y_train)

y2 = svc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy on unknown data is",accuracy_score(y_test,y2))

from sklearn.metrics import classification_report
print("Accuracy on unknown data is",classification_report(y_test,y2))

result = pd.DataFrame({'original' : y_test,'predicted' : y2})