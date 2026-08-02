#This module is used to train the model with the user image data

import json
import logging
import os

import cv2
import numpy as np
from django.conf import settings

import admin_state
import face_store

logger = logging.getLogger(__name__)

MODEL_PATH = 'trainer.yml'

# Which files went into the current model, per label. A count alone cannot say
# what is new, because the two enrolment paths name files differently -
# `12.jpg` from the camera and `img3.jpg` from an upload - so there is no
# ordering that reliably separates old from new.
MANIFEST = 'trained.json'


def load_model():
    """The trained recogniser, or None when nothing has been trained yet."""
    if not os.path.exists(MODEL_PATH):
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    return recognizer


def _read_crop(path):
    crop = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if crop is None:
        logger.warning("Skipping %s: not a readable image.", path)
        return None

    # Defensive: anything enrolled before the pipeline settled on a fixed size
    # still has to line up with the rest.
    size = settings.FACE_CROP_SIZE
    if crop.shape != (size, size):
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def _enrolled_files():
    """{label: {filename, ...}} for everyone currently enrolled."""
    return {label: set(os.listdir(folder))
            for label, folder in face_store.enrolled_folders()}


def _read_manifest():
    raw = admin_state.read(MANIFEST).strip()
    if not raw:
        return {}
    try:
        return {int(label): set(files) for label, files in json.loads(raw).items()}
    except (ValueError, AttributeError):
        logger.warning("%s is unreadable; treating the model as untrained.",
                       MANIFEST)
        return {}


def _write_manifest(files_by_label):
    admin_state.write(MANIFEST, json.dumps(
        {str(label): sorted(files) for label, files in files_by_label.items()}))
    admin_state.write(admin_state.TRAINED,
                      str(sum(len(f) for f in files_by_label.values())))


def _load(labels_and_files):
    """[(label, crop)] for the given {label: {filename}} selection."""
    samples, ids = [], []
    for label, filenames in labels_and_files.items():
        folder = face_store.folder_for('s%d' % label)
        for name in sorted(filenames):
            crop = _read_crop(os.path.join(folder, name))
            if crop is not None:
                samples.append(crop)
                ids.append(label)
    return samples, ids


def getImagesAndLabels():
    """Every stored crop with the label of the person it belongs to.

    Stored images are already the normalised crops that faces.py produces at
    enrolment, so there is nothing to detect here. The previous version ran a
    Haar cascade over each saved crop - detecting a face inside an image that
    was already a face - which quietly dropped any crop the cascade happened to
    miss, and normalised nothing.
    """
    return _load(_enrolled_files())


def begin(full=False):
    """Train the model, adding only what is new unless a rebuild is needed.

    LBPH can append to an existing model, which matters here because people
    enrol one at a time: rebuilding from every image on disk makes each new
    person cost more than the last. A full pass still happens when there is no
    model, no manifest, or when previously trained images have been removed -
    LBPH cannot forget, so anything subtractive means starting over.
    """
    current = _enrolled_files()
    if not current or not any(current.values()):
        # ValueError so the view can show this to the operator verbatim.
        raise ValueError(
            "No enrolled images were found, so there is nothing to train. "
            "Register at least one face first.")

    manifest = _read_manifest()
    model = None if full else load_model()

    removed = any(files - current.get(label, set())
                  for label, files in manifest.items())

    if model is None or not manifest or removed:
        reason = ("a full rebuild was requested" if full
                  else "no model yet" if model is None
                  else "no manifest" if not manifest
                  else "images were removed since the last run")
        logger.info("Training from scratch (%s).", reason)
        return _train_all(current)

    new = {label: files - manifest.get(label, set())
           for label, files in current.items()}
    new = {label: files for label, files in new.items() if files}

    if not new:
        logger.info("Model is already up to date; nothing to train.")
        return

    samples, ids = _load(new)
    if not samples:
        logger.info("Nothing readable to add; leaving the model as it is.")
        return

    model.update(samples, np.array(ids))
    model.write(MODEL_PATH)
    _write_manifest(current)
    logger.info("Added %d images for %d people to the existing model.",
                len(samples), len(new))


def _train_all(current):
    samples, ids = _load(current)
    if not samples:
        raise ValueError(
            "No enrolled images could be read, so there is nothing to train.")

    if len(set(ids)) < 2:
        logger.warning(
            "Only one person is enrolled. The recogniser will match everyone "
            "to them until a second person is added.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(samples, np.array(ids))
    recognizer.write(MODEL_PATH)
    _write_manifest(current)
    logger.info("Wrote %s from %d samples.", MODEL_PATH, len(samples))
