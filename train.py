import json
import os

import cv2
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras_preprocessing.image import img_to_array, np
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

base_model = InceptionV3(weights='imagenet', include_top=False)  # or use weights=None for from-scratch training

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

data = []
labels = []

yeses = [filename for filename in os.listdir('originals/yes') if filename.lower().endswith('.jpg')]
for filename in yeses:
    image = cv2.imread(f'originals/yes/{filename}')
    image = cv2.resize(image, (299, 299))  # 299x299 is the default Inception input size
    image = img_to_array(image)
    data.append(image)
    labels.append(1)  # 1 for yes firetruck

nos = [filename for filename in os.listdir('originals/no') if filename.lower().endswith('.jpg')]
for filename in nos:
    image = cv2.imread(f'originals/no/{filename}')
    image = cv2.resize(image, (299, 299))
    image = img_to_array(image)
    data.append(image)
    labels.append(0)  # 0 for no firetruck

data = np.array(data, dtype='float') / 255
labels = np.array(labels)

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25, random_state=24601)
train_y = to_categorical(train_y, num_classes=2)
test_y = to_categorical(test_y, num_classes=2)

# train the model on the new data for a few epochs
history1 = model.fit(train_x, train_y, batch_size=64, epochs=3, validation_data=(test_x, test_y))

with open('history_new_layers.json', 'w') as f:
    json.dump(history1.history, f)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy')

# Save our model each epoch along the way. If training stops providing a benefit, stop early to avoid overfitting
callbacks = [
    ModelCheckpoint(filepath='firetruck_{epoch:02d}.model'),
    EarlyStopping(min_delta=0.01, patience=5, mode='min', verbose=1)
]

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history2 = model.fit(train_x, train_y, batch_size=64, epochs=50, validation_data=(test_x, test_y), callbacks=callbacks)

model.save('firetruck.model')

with open('history_all_layers.json', 'w') as g:
    json.dump(history2.history, g)
