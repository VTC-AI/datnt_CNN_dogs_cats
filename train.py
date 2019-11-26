import keras
from model import classifier
from dataset import training_set, test_set


model = classifier

model.fit_generator(
  training_set,
  steps_per_epoch=(8000 // 32),
  epochs=25,
  validation_data=test_set,
  validation_steps=(2000 // 32)
)
model.save('model.h5')
