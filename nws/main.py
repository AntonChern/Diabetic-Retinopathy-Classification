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
    img_width, img_height = img_size, img_size
    input_shape = (img_width, img_height, 3)
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
        inception_net = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

        inception_net.trainable = False

        model = Sequential()
        model.add(inception_net)
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(grades))
        model.add(Activation(activation_function))

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
    print("Правильность на тестовых данных: %.2f%%" % (scores[1]*100))

    if isForTraining:
        model.save('C:/Users/Irusik/Desktop/' + saveName + '.h5')

if __name__ == "__main__":

    i_want_to_train = False

    if i_want_to_train:
        train(train_dir='trainBalancedBin',
              val_dir='valBalancedBin',
              test_dir='testBalancedBin',
              img_size=200,
              batch_size=16,
              nb_train_samples=4064,
              nb_validation_samples=800,
              nb_test_samples=800,
              isForTraining=i_want_to_train, isBinary=True, number_epochs=2, saveName='precision200')
    else:
        train(test_dir='testBalancedBin',
              img_size=200,
              batch_size=10,
              nb_test_samples=800,
              isForTraining=i_want_to_train, isBinary=True, loadName='precision200')

