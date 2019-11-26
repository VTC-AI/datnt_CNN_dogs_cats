from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras import backend as K
from keras.optimizers import RMSprop


opt = RMSprop(0.001)

input_shape = (3, 128, 128) if K.image_data_format() == 'chanels_first' else (128, 128, 3)

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(128, activation='relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(2, activation='softmax'))

classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
