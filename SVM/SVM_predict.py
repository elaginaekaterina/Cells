import pickle

import cv2
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

Categories = ['lymphocyte', 'monocyte', 'platelet']
flat_data_arr = []  # input array


#print(os.path.abspath(os.getcwd()))
model=pickle.load(open('img_model.p','rb'))


path1 = r'C:\Users\elagina\Documents\keras-tutorial\keras-tutorial\images\\'
for image in os.listdir(path1):
  img = imread(path1 + image)
  img_resize = resize(img, (150, 150, 3))
  l = [img_resize.flatten()]
  probability = model.predict_proba(l)

  # метка класса + вероятность на выходном изображении
  for y in y_pred:
    text = "{}: {:.2f}%".format(Categories, y_pred[0][i] * 100)
    cv2.putText(probability, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
              (0, 0, 255), 2)

  # показ выходного изображения
  cv2.imshow("Image", probability)
  cv2.waitKey(0)

  for ind, val in enumerate(Categories):
    print(f'{val} = {probability[0][ind] * 100}%')
    print("The predicted image is : "+Categories[model.predict(l)[0]])