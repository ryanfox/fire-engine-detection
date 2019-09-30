import sys

import cv2
import numpy as np

from keras.engine.saving import load_model


model = load_model('firetruck.model')

for filename in sys.argv[1:]:
    image = cv2.imread(filename)
    image = cv2.resize(image, (299, 299))
    image = image / 255

    # pass the image through the network to obtain our predictions
    preds = model.predict(np.expand_dims(image, axis=0))[0]
    arg = np.argmax(preds)
    prediction = ['No Firetruck', 'Firetruck'][arg]
    print(f'{filename[:12]:15}  {prediction:6}  {preds[arg] * 100}%')
