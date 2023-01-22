import argparse
import pickle
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.metrics import accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("-k", "--neighbors", type=int, default=100,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

Categories = ['lymphocyte', 'monocyte', 'platelet']
flat_data_arr = []  # input array
target_arr = []  # output array
datadir = r'C:\Users\elagina\Documents\keras-tutorial\keras-tutorial\cells_450'
# path which contains all the categories of images
for i in Categories:

    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)  # dataframe
df['Target'] = target
x = df.iloc[:, :-1]  # input data
y = df.iloc[:, -1]  # output data


model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

print('Splitted Successfully')
model.fit(x_train, y_train)
print('The Model is trained well with the given images')
# model.best_params_ contains the best parameters obtained from GridSearchCV

y_pred = model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")

pickle.dump(model, open('img_model.p', 'wb'))
print("Pickle is dumped successfully")

# print(os.path.abspath(os.getcwd()))
model = pickle.load(open('img_model.p', 'rb'))

from PIL import Image, ImageDraw, ImageFont

path1 = r'C:\Users\elagina\Documents\keras-tutorial\keras-tutorial\images\\'
for image in os.listdir(path1):
    img = imread(path1 + image)
    img_resize = resize(img, (150, 150, 3))
    l = [img_resize.flatten()]
    probability = model.predict_proba(l)

    for ind, val in enumerate(Categories):
        print(f'{val} = {probability[0][ind] * 100}%')
        print("The predicted image is : " + Categories[model.predict(l)[0]])
        text = "{}: {:.2f}%".format(Categories[model.predict(l)[0]], probability[0][ind] * 100)
        output = img.copy()
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.imshow("Image", output)
        cv2.waitKey(0)