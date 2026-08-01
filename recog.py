#This module is used to train the model with the user image data

import logging
import os
import re

import cv2
import numpy as np
from django.conf import settings
from PIL import Image

import admin_state

logger = logging.getLogger(__name__)

CASCADE_PATH = 'haarcascade_frontalface_default.xml'
MODEL_PATH = 'trainer.yml'


def _face_folders():
    """Every enrolled person's folder, as (label, path) pairs.

    Matching on s<number> is what lets static/ and staticfiles/ stop being
    special cases. The previous version skipped them with startswith('st') and
    startswith('sm') tests, which would also have silently skipped a
    legitimately named folder had one ever started with those letters.
    """
    for entry in sorted(os.listdir(settings.BASE_DIR)):
        if not re.match(settings.FACE_USERNAME_PATTERN, entry):
            continue
        path = os.path.join(settings.BASE_DIR, entry)
        if os.path.isdir(path):
            yield int(entry[1:]), path


def getImagesAndLabels(detector):
    face_samples = []
    ids = []
    images_read = 0

    for label, folder in _face_folders():
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            images_read += 1

            try:
                grey = np.array(Image.open(image_path).convert('L'), 'uint8')
            except OSError:
                # One unreadable file should not abort a long training run.
                logger.warning("Skipping %s: not a readable image.", image_path)
                continue

            for (x, y, w, h) in detector.detectMultiScale(grey):
                face_samples.append(grey[y:y + h, x:x + w])
                ids.append(label)

    admin_state.write(admin_state.TRAINED, str(images_read))
    logger.info("Read %d images and found %d faces.", images_read, len(face_samples))
    return face_samples, ids


def begin():
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    faces, ids = getImagesAndLabels(detector)

    if not faces:
        # ValueError so the view can show this to the operator verbatim.
        raise ValueError(
            "No faces were found in the enrolled images, so there is nothing "
            "to train. Register at least one face first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write(MODEL_PATH)
    logger.info("Wrote %s from %d face samples.", MODEL_PATH, len(faces))
