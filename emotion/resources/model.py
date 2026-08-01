import os

# model.json and model_weights.h5 are Keras 2 artifacts. TensorFlow 2.16
# switched tf.keras to Keras 3, which cannot load them, so select the legacy
# implementation - this has to happen before tensorflow is imported.
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf

class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    def __init__(self):
        with open('emotion/resources/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights('emotion/resources/model_weights.h5')

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
