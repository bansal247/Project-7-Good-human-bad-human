from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator


class Model:
    def __init__(self):
        self.test_set = None
        self.validation_set = None
        self.training_set = None
        self.classifier = Sequential()
        self.classifier.add(Conv2D(32, (3, 3), input_shape=(247, 326, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(3, 3)))
        self.classifier.add(Conv2D(16, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(3, 3)))
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=256, activation='relu'))
        self.classifier.add(Dropout(0.1))
        self.classifier.add(Dense(units=128, activation='relu'))
        self.classifier.add(Dropout(0.1))
        self.classifier.add(Dense(units=10, activation='relu'))
        self.classifier.add(Dense(units=1, activation='sigmoid'))

        self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def data_generation(self, path_to_training_data, path_to_validation_data, path_to_test_data):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255, )
        valid_datagen = ImageDataGenerator(rescale=1. / 255, )
        self.training_set = train_datagen.flow_from_directory(path_to_training_data,
                                                              target_size=(247, 326),
                                                              batch_size=64,
                                                              class_mode='binary')
        self.validation_set = valid_datagen.flow_from_directory(path_to_validation_data,
                                                                target_size=(247, 326),
                                                                batch_size=64,
                                                                class_mode='binary')
        self.test_set = test_datagen.flow_from_directory(path_to_test_data,
                                                         target_size=(247, 326),
                                                         batch_size=64,
                                                         class_mode='binary')

    def fit(self):
        self.classifier.fit_generator(self.training_set,
                                      epochs=19,
                                      validation_data=self.validation_set,
                                      validation_steps=1)
        self.classifier.save("model.h5")
