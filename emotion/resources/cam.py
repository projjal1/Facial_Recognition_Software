import logging
import os

import cv2
import numpy as np

from emotion.resources.model import FacialExpressionModel

logger = logging.getLogger(__name__)

# Relative to this file rather than the working directory, so the cascade is
# found regardless of where the process was started.
RESOURCES = os.path.dirname(os.path.abspath(__file__))

facec = cv2.CascadeClassifier(
    os.path.join(RESOURCES, 'haarcascade_frontalface_default.xml'))
model = FacialExpressionModel()
font = cv2.FONT_HERSHEY_SIMPLEX


def frames(source):
    """Yield frames from `source` annotated with the predicted emotion.

    A generator rather than a loop owning a cv2 window, for the same reason as
    recognition.frames(): the caller decides what to do with each frame, which
    is what lets this feed an HTTP response instead of a desktop session.
    """
    for fr in source:
        if fr is None:
            continue

        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = cv2.resize(gray_fr[y:y + h, x:x + w], (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        yield fr
