"""Face detection and crop normalisation, shared by every path that stores or
matches a face.

Both halves of the pipeline have to agree. LBPH compares histograms of local
texture, so if images are enrolled at one scale and matched at another - or one
path equalises lighting and the other does not - accuracy falls in a way that
reads as a badly chosen threshold rather than as a preprocessing mismatch.
Keeping detection and normalisation in one module is what makes that agreement
automatic instead of a thing to remember.

The detector is the SSD already vendored for the mask app, not the Haar cascade
the project started with. It is markedly better on angled faces, glasses and
poor lighting, and it costs no new dependency.
"""

import logging
import os

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

_net = None


def _detector():
    """The SSD face detector, loaded once and reused."""
    global _net
    if _net is None:
        resources = os.path.join(settings.BASE_DIR, 'mask', 'resources')
        _net = cv2.dnn.readNet(
            os.path.join(resources, 'deploy.prototxt'),
            os.path.join(resources, 'res10_300x300_ssd_iter_140000.caffemodel'))
        logger.info("Loaded the SSD face detector.")
    return _net


def detect(frame):
    """Faces in a BGR frame as (x, y, w, h), largest first.

    Anything below FACE_MIN_SIZE is dropped: a face that small carries no usable
    identity signal and is mostly a source of false positives. The old Haar call
    accepted faces down to 10 pixels.
    """
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    net = _detector()
    net.setInput(blob)
    detections = net.forward()

    minimum = settings.FACE_MIN_SIZE
    boxes = []

    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] < settings.FACE_DETECTOR_CONFIDENCE:
            continue

        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        x1, y1, x2, y2 = box.astype('int')
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)

        w, h = x2 - x1, y2 - y1
        if w < minimum or h < minimum:
            continue
        boxes.append((x1, y1, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def normalise(frame, box):
    """The fixed-size, contrast-equalised grey crop that gets stored and matched.

    Fixed size because LBPH divides a crop into a grid and histograms each cell;
    feeding it crops of varying size distorts that grid. CLAHE rather than a
    global histogram equalisation because it lifts local contrast without
    amplifying noise across the whole crop, which is what side lighting needs.
    """
    x, y, w, h = box
    grey = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

    size = settings.FACE_CROP_SIZE
    grey = cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)

    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)


def is_sharp(crop):
    """Whether a crop carries enough detail to be worth enrolling.

    Variance of the Laplacian: low variance means few strong edges, which for a
    face crop means motion blur or a soft focus. Enrolling those teaches the
    model a smeared version of someone.
    """
    return cv2.Laplacian(crop, cv2.CV_64F).var() >= settings.FACE_BLUR_THRESHOLD


def crops(frame):
    """Yield (box, normalised crop) for each face in a BGR frame."""
    for box in detect(frame):
        yield box, normalise(frame, box)
