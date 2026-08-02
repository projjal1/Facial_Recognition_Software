#This module is used to train the model with the user image data

import logging
import os

import cv2
import numpy as np
from django.conf import settings

import admin_state
import face_store

logger = logging.getLogger(__name__)

MODEL_PATH = 'trainer.yml'


def getImagesAndLabels():
    """Every stored crop with the label of the person it belongs to.

    Stored images are already the normalised crops that faces.py produces at
    enrolment, so there is nothing to detect here. The previous version ran a
    Haar cascade over each saved crop - detecting a face inside an image that
    was already a face - which quietly dropped any crop the cascade happened to
    miss, and normalised nothing.
    """
    samples = []
    ids = []
    images_read = 0
    size = settings.FACE_CROP_SIZE

    for label, folder in face_store.enrolled_folders():
        for image_name in os.listdir(folder):
            path = os.path.join(folder, image_name)
            images_read += 1

            crop = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if crop is None:
                # One unreadable file should not abort a long training run.
                logger.warning("Skipping %s: not a readable image.", path)
                continue

            # Defensive: anything enrolled before the pipeline settled on a
            # fixed size still has to line up with the rest.
            if crop.shape != (size, size):
                crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

            samples.append(crop)
            ids.append(label)

    admin_state.write(admin_state.TRAINED, str(images_read))
    logger.info("Read %d images across %d people.", images_read, len(set(ids)))
    return samples, ids


def begin():
    samples, ids = getImagesAndLabels()

    if not samples:
        # ValueError so the view can show this to the operator verbatim.
        raise ValueError(
            "No enrolled images were found, so there is nothing to train. "
            "Register at least one face first.")

    if len(set(ids)) < 2:
        logger.warning(
            "Only one person is enrolled. The recogniser will match everyone "
            "to them until a second person is added.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(samples, np.array(ids))
    recognizer.write(MODEL_PATH)
    logger.info("Wrote %s from %d samples.", MODEL_PATH, len(samples))
