from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import InceptionV3
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.models import load_model
import torch.optim as optim
import torch
import sklearn
from sklearn.metrics import recall_score

import shutil
import os
import pandas as pd
import xlrd

def train(test_dir, img_size, batch_size, nb_test_samples, isForTraining, isBinary, train_dir='', val_dir='',
          nb_train_samples=0, nb_validation_samples=0, number_epochs=5, saveName='', loadName=''):
    # Каталог с данными для обучения
    #train_dir = 'trainM2bin'
    # Каталог с данными для проверки
    #val_dir = 'valM2bin'
    # Каталог с данными для тестирования
    #test_dir = 'testM2bin'
    # Размеры изображения
    img_width, img_height = img_size, img_size
    # Размерность тензора на основе изображения для входных данных в нейронную сеть
    # backend Tensorflow, channels_last
    input_shape = (img_width, img_height, 3)
    # Размер мини-выборки
    #batch_size = 8
    # Количество изображений для обучения
    # nb_train_samples = 1022
    # Количество изображений для проверки
    # nb_validation_samples = 216
    # Количество изображений для тестирования
    # nb_test_samples = 216

    # model = load_model('C:/Users/Irusik/Desktop/CroppedAndBalancedAllWeights.h5')
    datagen = ImageDataGenerator(rescale=1. / 255)

    if isBinary:
        grades = 1
        activation_function = 'sigmoid'
        classification_type = 'binary'
    else:
        grades = 5
        activation_function = 'softmax'
        classification_type = 'categorical'

    if isForTraining:
        # inception_net = InceptionV3(include_top=False, input_shape=(150, 150, 3))
        inception_net = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

        inception_net.trainable = False

        # inception_net.summary()

        model = Sequential()
        # Добавляем в модель сеть VGG16 вместо слоя
        model.add(inception_net)
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(grades))
        model.add(Activation(activation_function))

        # model.summary()
        model.compile(loss=classification_type + '_crossentropy',
                      optimizer=Adam(lr=1e-5),
                      metrics=['accuracy'])

        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=classification_type)

        val_generator = datagen.flow_from_directory(
            val_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=classification_type)

        test_generator = datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=classification_type)

        model.fit(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=number_epochs,
            validation_data=val_generator,
            validation_steps=nb_validation_samples // batch_size)
    else:
        model = load_model('C:/Users/Irusik/Desktop/' + loadName + '.h5')

        if isBinary:
            grades += 1

        for current_grade in range(grades):
            size = 0

            for grade in range(grades):
                if grade == current_grade:
                    size = len(os.listdir(test_dir + "/" + str(grade)))
                    continue
                for img_name in os.listdir(test_dir + "/" + str(grade)):
                    os.rename(test_dir + "/" + str(grade) + "/" + img_name, str(grade) + "/" + img_name)

            test_generator = datagen.flow_from_directory(
                test_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=classification_type)

            scores = model.evaluate_generator(test_generator, size // batch_size)

            print("Точность класса " + str(current_grade) + ": %.2f%%" % (scores[1]*100))

            for grade in range(grades):
                if grade == current_grade:
                    continue
                for img_name in os.listdir(str(grade)):
                    os.rename(str(grade) + "/" + img_name, test_dir + "/" + str(grade) + "/" + img_name)

        test_generator = datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=classification_type)

    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

    print(scores)
    print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

    if isForTraining:
        model.save('C:/Users/Irusik/Desktop/' + saveName + '.h5')

    # inception_net.trainable = True
    # trainable = False
    # for layer in inception_net.layers:
    #     if layer.name == 'block5_conv1':
    #         trainable = True
    #     layer.trainable = trainable

    #Проверяем количество обучаемых параметров
    # model.summary()
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=Adam(lr=1e-5),
    #               metrics=['accuracy'])
    #
    # model.fit(
    #     train_generator,
    #     steps_per_epoch=nb_train_samples // batch_size,
    #     epochs=2,
    #     validation_data=val_generator,
    #     validation_steps=nb_validation_samples // batch_size)
    #
    # scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    # print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

if __name__ == "__main__":
    # indices = [0, 0, 0, 0, 0]
    # formatFile = ".jpeg"
    # shift = 0
    # folder = ""
    # rb = xlrd.open_workbook("C:/Users/Irusik/Desktop/dataset/train/labels.xls", formatting_info=True)
    # sheet = rb.sheet_by_index(0)
    # for rownum in range(1, sheet.nrows):
    #     row = sheet.row_values(rownum)
    #     rowString = row[shift]
    #     grade = int(rowString[-1])
    #     number = indices[grade]
    #     indices[grade] += 1
    #     name = str(rowString[2 * (1 + len(str(rownum - 1))):-2])
    #     if grade == 0:
    #         if number < 3871:
    #             folder = "test"
    #         elif number > 21938:
    #             folder = "val"
    #         else:
    #             folder = "train"
    #     elif grade == 1:
    #         if number < 366:
    #             folder = "test"
    #         elif number > 2076:
    #             folder = "val"
    #         else:
    #             folder = "train"
    #     elif grade == 2:
    #         if number < 790:
    #             folder = "test"
    #         elif number > 4501:
    #             folder = "val"
    #         else:
    #             folder = "train"
    #     elif grade == 3:
    #         if number < 130:
    #             folder = "test"
    #         elif number > 742:
    #             folder = "val"
    #         else:
    #             folder = "train"
    #     elif grade == 4:
    #         if number < 100:
    #             folder = "test"
    #         elif number > 607:
    #             folder = "val"
    #         else:
    #             folder = "train"
    #     print(name + " -> " + folder + "/" + str(grade) + "/R" + str(grade) + "_" + str(number) + formatFile)
        # os.rename("C:/Users/Irusik/Desktop/dataset/train/resized_train_cropped/" + name + formatFile,
        #           folder + "/" + str(grade) + "/R" + str(grade) + "_" + str(number) + formatFile)
    # print(indices)

    # indices = [0, 0, 0, 0, 0]
    # shift = 0
    # rb = xlrd.open_workbook("messidor2.xls", formatting_info=True)
    # sheet = rb.sheet_by_index(0)
    # for rownum in range(0, sheet.nrows, 4):
    #     row = sheet.row_values(rownum)
    #     grade = sheet.row_values(rownum + 1)
    #     gradeIndex = int(grade[shift])
    #     number = indices[gradeIndex]
    #     indices[gradeIndex] += 1
    #     name = str(row[shift])
    #     os.rename("IMAGES/" + name, "IMAGES_MESSIDOR2/" + str(gradeIndex) + "/R" + str(gradeIndex) + "_" + str(number) + name[-4:])

    # i0 = 0
    # i1 = 0
    # i2 = 0
    # i3 = 0
    # for jindex in range(1, 4):
    #     for index in range(1, 5):
    #         name = "Base" + str(jindex) + str(index)
    #         rb = xlrd.open_workbook(name + "/Annotation " + name + ".xls", formatting_info=True)
    #         sheet = rb.sheet_by_index(0)
    #         s = 0
    #         for rownum in range(1, sheet.nrows):
    #             row = sheet.row_values(rownum)
    #             if row[2] == 0:
    #                 s = i0
    #                 i0 += 1
    #             elif row[2] == 1:
    #                 s = i1
    #                 i1 += 1
    #             elif row[2] == 2:
    #                 s = i2
    #                 i2 += 1
    #             elif row[2] == 3:
    #                 s = i3
    #                 i3 += 1
    #             os.rename(name + "/" + str(row[0]), "Base/" + str(int(row[2])) + "/R" + str(int(row[2])) + "_" + str(s) + ".tif")

    # i_want_to_train = True
    i_want_to_train = False

    # inception_net = InceptionV3(weights='imagenet', include_top=True, input_shape=(299, 299, 3))
    # inception_net.summary()
    # print("===================================================================================================================================")
    # inception_net2 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    # inception_net2.summary()

    # define actual
    act_pos1 = [1 for _ in range(100)]
    act_pos2 = [2 for _ in range(100)]
    act_neg = [0 for _ in range(10000)]
    y_true = act_pos1 + act_pos2 + act_neg
    # define predictions
    pred_pos1 = [1 for _ in range(90)] + [2 for _ in range(5)] + [0 for _ in range(5)]
    pred_pos2 = [2 for _ in range(90)] + [1 for _ in range(10)]
    pred_neg = [0 for _ in range(9990)] + [1 for _ in range(5)] + [2 for _ in range(5)]
    y_pred = pred_pos1 + pred_pos2 + pred_neg
    # calculate recall
    recall = recall_score(y_true, y_pred, labels=[1, 2], average='micro')
    print('Recall: %.3f' % recall)

    # if i_want_to_train:
    #     train(train_dir='trainBalancedBin',
    #           val_dir='valBalancedBin',
    #           test_dir='testBalancedBin',
    #           img_size=200,
    #           batch_size=16,
    #           nb_train_samples=4064,
    #           nb_validation_samples=800,
    #           nb_test_samples=800,
    #           isForTraining=i_want_to_train, isBinary=True, number_epochs=2, saveName='precision200')
    # else:
    #     train(test_dir='testBalancedBin',
    #           img_size=200,
    #           batch_size=10,
    #           nb_test_samples=800,
    #           isForTraining=i_want_to_train, isBinary=True, loadName='precision200')

