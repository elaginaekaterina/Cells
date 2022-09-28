# импорт необходимых пакетов
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

# определение класса SmallVGGNet и метода сборки (build)
class SmallVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # инициализация модели и размера входного изображения
        # для порядка каналов “channel_last” и размер канала
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # при использовании порядка "channels first", обновляем
        # входное изображение и размер канала
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

# слои CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same",
        input_shape=inputShape)) # Первый слой CONV имеет 32 фильтра размером 3х3
        model.add(Activation("relu")) #  функция активации
        model.add(BatchNormalization(axis=chanDim)) # пакетная нормализация
        model.add(MaxPooling2D(pool_size=(2, 2))) # функция максимума
        model.add(Dropout(0.25)) # метод исключения

 # слои (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

# слои (CONV => RELU) * 3 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

# слои (CONV => RELU) * 3 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

# набор слоев FC => RELU
        model.add(Flatten())
        model.add(Dense(512)) # полностью связанные слои
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

# классификатор softmax возвращает вероятность принадлежности
        # к определённому классу для каждой метки
        model.add(Dense(classes))
        model.add(Activation("softmax"))

# возвращаем собранную архитектуру нейронной сети
        return model