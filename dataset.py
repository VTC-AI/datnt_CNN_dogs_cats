from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
  rescale=1./255,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
  'dataset/training_set',
  target_size=(128, 128),
  batch_size=32,
  class_mode='categorical',
  seed=42
)

test_set = test_datagen.flow_from_directory(
  'dataset/test_set',
  target_size=(128, 128),
  batch_size=32,
  class_mode='categorical',
  seed=42
)
