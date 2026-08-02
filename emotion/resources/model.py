import os

# model.json and model_weights.h5 are Keras 2 artifacts. TensorFlow 2.16
# switched tf.keras to Keras 3, which cannot load them, so select the legacy
# implementation - this has to happen before tensorflow is imported.
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

from tensorflow.keras.models import model_from_json
import numpy as np

# Relative to this file rather than the working directory, so the model is
# found regardless of where the process was started.
RESOURCES = os.path.dirname(os.path.abspath(__file__))


class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self):
        with open(os.path.join(RESOURCES, 'model.json'), "r") as json_file:
            self.loaded_model = model_from_json(json_file.read())

        # load weights into the new model
        self.loaded_model.load_weights(os.path.join(RESOURCES, 'model_weights.h5'))

    def predict_emotion(self, img):
        # Called directly rather than through .predict(). For a single 48x48
        # sample, predict() spends far longer building a data adapter, a
        # tf.function and a progress logger than the forward pass itself takes -
        # which is most of why this feed lagged where the others did not.
        preds = self.loaded_model(np.asarray(img, dtype='float32'), training=False)
        return FacialExpressionModel.EMOTIONS_LIST[int(np.argmax(preds))]
