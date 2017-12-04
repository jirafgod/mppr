import numpy
import requests
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import *
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils, plot_model
from keras.callbacks import TensorBoard
import numpy as np
import time

# Загружаем данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование размерности изображений
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
# Нормализация данных
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Преобразуем метки в категории
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Загружаем данные об архитектуре сети из файла json
if os.path.exists("mnist_model.json"):
    print("Модель уже есть\n")
    json_file = open("mnist_model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    # Создаем модель на основе загруженных данных
    model = model_from_json(loaded_model_json)
    # Загружаем веса в модель
    model.load_weights("mnist_model.h5")

    # Компилируем модель
    payload = {'test': time.gmtime()}
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    imgs = requests.get('http://vkgamebot.com/mppr/test.txt', params=payload).text
    imgs = imgs.split(',')
    data = np.zeros((1, 784))
    i = 0
    while i < 784:
        data[0][i] = imgs[i]
        i = i + 1
    encoded_imgs = model.predict(data, batch_size=1)
    pos = 0
    max = encoded_imgs[0][0]
    for i in range(len(encoded_imgs[0])):
        if encoded_imgs[0][i] > max:
            max = encoded_imgs[0][i]
            pos = i
    print(pos)
else:
    print("Модели нет, все по новой\n")
    # Устанавливаем seed для повторяемости результатов
    numpy.random.seed(42)

    # Создаем последовательную модель
    model = Sequential()

    # Добавляем уровни сети
    model.add(Dense(300, input_dim=784, activation="relu", kernel_initializer="normal"))
    model.add(Dropout(0.2))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Компилируем модель
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    plot_model(model, to_file='model.png', show_shapes=True)

    print(model.summary())

    # Обучаем сеть
    model.fit(X_train, Y_train, batch_size=200, epochs=10, validation_split=0.2, verbose=2)

    # Оцениваем качество обучения сети на тестовых данных
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
    # Генерируем описание модели в формате json
    model_json = model.to_json()
    # Записываем модель в файл
    json_file = open("mnist_model.json", "w")
    json_file.write(model_json)
    json_file.close()
    model.save_weights("mnist_model.h5")
