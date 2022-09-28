# импорт бэкендa Agg из matplotlib для сохранения графиков на диск
import matplotlib
matplotlib.use("Agg")

# подключение необходимых пакетов
from pyimagesearch.smallvggnet import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# создание парсера аргументов и их передача
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
ap.add_argument("-k", "--matrix", required=True,
	help="path to output matrix")
args = vars(ap.parse_args())

# инициализация данных и меток
print("[INFO] loading images...")
data = []
labels = []

# рандомно перемешиваем изображения
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# цикл по изображениям
for imagePath in imagePaths:
	# загрузка изображений, изменение размера на 64x64 пикселей,
	# изменённое изображение добавляем в список
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	data.append(image)

	# извлечение метки класса из пути к изображению и обновление
	# списока меток
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# масштаб интенсивности пикселей в диапазон [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# деление данных на обучающую и тестовую выборки, используя 75%
# данных для обучения и оставшиеся 25% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# конвертация меткок из целых чисел в векторы
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# создание генератора для добавления изображений
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# инициализация VGG-подобной сверточной нейросети
model = SmallVGGNet.build(width=64, height=64, depth=3,
	classes=len(lb.classes_))

# инициализация скорости обучения, общего числа эпох
# и размера пакета
INIT_LR = 0.5
EPOCHS = 55
BS = 32

# компиляция модели с помощью SGD
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# обучение нейросети
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# оценка нейросети
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# построение матрицы различий
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()

y_pred = np.argmax(predictions, axis=1)
y_test = np.argmax(testY, axis=1)

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar=False,
			xticklabels = labels, yticklabels = labels)
plt.xlabel('Прогнозируемое значение')
plt.ylabel('Настоящее значение')
plt.title('Матрица различий')
plt.savefig(args["matrix"])
print(confusion_matrix(y_test, y_pred))

# построение графиков потерь и точности
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="Потери при обучении")
plt.plot(N, H.history["val_loss"], label="Потери при оценке")
plt.plot(N, H.history["accuracy"], label="Точность обучения")
plt.plot(N, H.history["val_accuracy"], label="Точнось оценивания")
plt.ylim(0, 10)
plt.title("Потери при обучении и точность обучения(Small VGG)")
plt.xlabel("Эпохи#")
plt.ylabel("Потери/Точность")
plt.legend()
plt.savefig(args["plot"])

# сохранение модели и бинаризатора меток на диск
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()