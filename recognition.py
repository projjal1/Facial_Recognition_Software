"""The recognition loop shared by the local and remote capture paths.

Frames are yielded rather than drawn, so the same logic feeds an HTTP response
instead of a window on the server.

Identity is decided per tracked face. Each frame casts one vote for what that
track looks like, and an entry is written when one identity has enough votes -
not when a global counter reaches a number, which is how a crowd used to
produce a log entry naming whoever was in the final frame.
"""

import datetime
import logging
import os
import time

import cv2
from django.conf import settings

import admin_state
import alerts
import faces
import tracking

logger = logging.getLogger(__name__)

MODEL_PATH = 'trainer.yml'

# After logging someone, wait this long before logging them again. Tracked per
# face rather than globally, so one person being logged does not mute another.
LOG_COOLDOWN_SECONDS = 3


def _load_recognizer(threshold):
    if not os.path.exists(MODEL_PATH):
        # ValueError so the view can show this text to the operator directly.
        raise ValueError(
            "No trained model found. Train the model before starting detection.")

    # With a threshold set, predict() returns -1 rather than the nearest label
    # when nothing is close enough. That is what makes "not recognised" a real
    # answer instead of something inferred from the distance afterwards.
    recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=threshold)
    recognizer.read(MODEL_PATH)
    return recognizer


def _log_entry(name):
    stamp = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    admin_state.append(admin_state.LOGS, "\n %s logged at %s" % (name, stamp))
    logger.info("Logged entry for %s.", name)


def _send_alert(number):
    logger.warning("Unrecognised face persisted; sending alert.")
    try:
        alerts.alert(number)
    except Exception:
        # An alert that fails to send must not take the capture loop down.
        logger.exception("Could not send the alert SMS.")


def frames(source, names, threshold):
    """Yield annotated frames, logging entries and raising alerts as it goes.

    `names` maps an LBPH label to a username. An unknown label - including the
    -1 the recogniser returns when nothing is close enough - counts as a vote
    for "not recognised" rather than being attributed to someone.
    """
    recognizer = _load_recognizer(threshold)
    tracker = tracking.Tracker()
    font = cv2.FONT_HERSHEY_SIMPLEX

    alert_number = admin_state.read(admin_state.MOBILE_NO).strip()
    votes_to_log = settings.FACE_FRAMES_TO_LOG
    votes_to_alert = settings.FACE_FRAMES_TO_ALERT

    detect_every = max(1, settings.FACE_DETECT_EVERY)
    boxes = []
    index = 0

    for img in source:
        if img is None:
            continue

        # Detection dominates the per-frame cost. Faces do not move far between
        # frames, so reusing the last boxes for a frame or two keeps the video
        # at the camera's rate without changing what gets recognised. Crops are
        # still taken from the current frame, and every frame still votes.
        if index % detect_every == 0:
            boxes = faces.detect(img)
        index += 1

        detections = [(box, faces.normalise(img, box)) for box in boxes]

        for (track, box), (_, crop) in zip(tracker.update(boxes), detections):
            label, _confidence = recognizer.predict(crop)
            track.record(names.get(label))

            leader, count = track.leader()
            now = time.monotonic()

            if leader is not None and count >= votes_to_log:
                if now >= track.cooldown_until:
                    _log_entry(leader)
                    track.cooldown_until = now + LOG_COOLDOWN_SECONDS
                    caption = "Logged to system"
                else:
                    caption = "Logged - waiting before logging again"
                track.reset()
            elif leader is None and count >= votes_to_alert:
                _send_alert(alert_number)
                track.reset()
                caption = "Unrecognised face - system alerted"
            elif leader is None:
                caption = "Detecting.."
            else:
                caption = "Detected " + leader

            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, caption, (x + 5, y - 5), font, 0.7,
                        (255, 255, 255), 2)

        yield img
