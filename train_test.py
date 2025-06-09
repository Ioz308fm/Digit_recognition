import os
import cv2
import numpy as np
import tensorflow as tf
import keras
import json
import math
from scipy import ndimage
from scipy.ndimage import center_of_mass
import pandas as pd

from functions_py import qwen_pipeline_train, qwen_optimized_test_class_alg, batch_get_labels

# тренировка новой модели
train_new_model = False

if train_new_model:
    # загрузка MNIST и разбиение на тестовые и тренировочные данные
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # нормирование данных
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # создание модели
    # плоский входной слой для пикселей
    # два скрытых полносвязных слоя
    # один односвязный выходной слой для 10 чисел
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    # компиляция и оптимизация модели
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # тренировка модели
    model.fit(X_train, y_train, epochs=3)
    # оценка модели
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # сохранение
    model.save("checkpoint/handwritten_digits.keras")
else:
    model = keras.saving.load_model("checkpoint/handwritten_digits.keras")

def getBestShift(img):
    cy, cx = center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def rec_digit(img):
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gray = 255 - img
    # применяем пороговую обработку
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # удаляем нулевые строки и столбцы
    while np.sum(gray[0]) == 0:
        gray = gray[1:]
    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)
    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]
    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)
    rows, cols = gray.shape

    # изменяем размер, чтобы помещалось в box 20x20 пикселей
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        gray = cv2.resize(gray, (cols, rows))

    # расширяем до размера 28x28
    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.pad(gray, (rowsPadding, colsPadding), 'constant')

    # сдвигаем центр масс
    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted

    #cv2.imwrite('test numbers/digit{}.png'.format(img_number), gray)
    img = gray / 255.0
    img = np.array(img).reshape(-1, 28, 28, 1)
    print('DDDD')
    print(model.predict(img))
    return model.predict(img)

def rec_digit_2(img, data_train, data_train_means, data_test, fourier_coef):
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # gray = 255 - img
    # dim = (28, 28)
    # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #cv2.imwrite('test numbers/digit{}.png'.format(img_number), gray)
    # img = resized / 255.0
    # print('shape')
    # print(img.shape)
    # img = np.array(img).reshape(-1, 28, 28, 1)
    # flattened_image = img.reshape(1, -1)  # форма: (1, 784)
    print(img.shape)
    
    df = img.copy()
# Преобразуем в DataFrame
    df.columns = [f"pixel{i}" for i in range(784)]

    img_transformed = transform_pixels_to_sums_vectorized(df.copy()).values


    t = 1.5847369

    target = 'label'
    res = batch_get_labels(
        target, data_train, data_train_means, fourier_coef, img_transformed, t, is_labeled = False
    )
    print('REEEEES')
    print(res)
    print(res.argmax)
    return res


def transform_pixels_to_sums_vectorized(df):
    # Создаем копию исходного DataFrame
    result_df = pd.DataFrame()
    
    # Получаем все пиксельные колонки
    pixel_cols = [f'pixel{i}' for i in range(784)]
    pixels = df[pixel_cols].values
    
    # Преобразуем в 3D массив (n_images, 28, 28)
    images = pixels.reshape(-1, 28, 28)
    
    # Суммы по строкам (axis=2 - сумма по столбцам в каждом ряду)
    row_sums = images.sum(axis=2)
    
    # Суммы по столбцам (axis=1 - сумма по рядам в каждом столбце)
    col_sums = images.sum(axis=1)
    
    # Добавляем суммы в DataFrame
    for i in range(28):
        result_df[f'sum_row{i}'] = row_sums[:, i] / 28
    for i in range(28):
        result_df[f'sum_column{i}'] = col_sums[:, i] / 28

    return result_df
