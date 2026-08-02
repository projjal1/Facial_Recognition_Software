import logging

import cv2
import numpy as np
from django.conf import settings

import faces
from emotion.resources.model import FacialExpressionModel

logger = logging.getLogger(__name__)

model = FacialExpressionModel()
font = cv2.FONT_HERSHEY_SIMPLEX


def frames(source):
    """Yield frames from `source` annotated with the predicted emotion.

    A generator rather than a loop owning a cv2 window, for the same reason as
    recognition.frames(): the caller decides what to do with each frame, which
    is what lets this feed an HTTP response instead of a desktop session.

    Detection and the classifier run every Nth frame and the result is drawn on
    the ones in between - an expression does not change faster than that.
    """
    detect_every = max(1, settings.FACE_DETECT_EVERY)
    results = []
    index = 0

    for fr in source:
        if fr is None:
            continue

        if index % detect_every == 0:
            grey = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            results = []

            # The shared SSD, not the Haar cascade this used to run. Haar was
            # sweeping the whole frame at every scale from 24px up, with no
            # minimum size, which cost more than everything else here combined.
            # The SSD is a fixed 300x300 forward pass whatever the frame size.
            for (x, y, w, h) in faces.detect(fr):
                # Deliberately not faces.normalise(): that equalises and resizes
                # for the recogniser, and this model was trained on plain 48x48
                # greyscale crops.
                roi = cv2.resize(grey[y:y + h, x:x + w], (48, 48))
                pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                results.append(((x, y, w, h), pred))
        index += 1

        for (x, y, w, h), pred in results:
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        yield fr
