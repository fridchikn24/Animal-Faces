import keras.preprocessing.image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
#import cv2


path = Path("../afhq")



training_generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range=7,
                                        horizontal_flip=True,
                                        zoom_range=0.2)
training_dataset = training_generator.flow_from_directory(path /'train',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'categorical',
                                                        shuffle = True)

print(training_dataset.class_indices)

test_generator = ImageDataGenerator(rescale=1./255)
test_dataset = test_generator.flow_from_directory(path /'val',
                                                     target_size = (64, 64),
                                                     batch_size = 1,
                                                     class_mode = 'categorical',
                                                     shuffle = False)

cnn = Sequential()
cnn.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation='relu'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Conv2D(32, (3,3), activation='relu'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Flatten())
cnn.add(Dense(units = 731, activation='relu'))
cnn.add(Dense(units = 64, activation='relu'))
cnn.add(Dense(units = 3, activation='softmax'))

cnn.summary()

cnn.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])
historic = cnn.fit(training_dataset, epochs = 10)

forecasts = cnn.predict(test_dataset)
forecasts = np.argmax(forecasts, axis = 1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_dataset.classes, forecasts)
print(cm)


from sklearn.metrics import classification_report
print(classification_report(test_dataset.classes, forecasts))


