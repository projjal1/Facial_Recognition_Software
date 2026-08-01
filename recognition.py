"""The recognition loop shared by the local and remote capture paths.

identify.py and webcam.py differ only in where frames come from, but each used
to carry its own near-identical copy of this logic. That is how their
confidence thresholds drifted apart - 53 in one, 48 in the other - with no
record of why. Keeping the loop in one place means a tuning change applies to
both, and the frame source is now the only thing either module supplies.
"""

import datetime
import logging
import os
import time

import cv2
from django.conf import settings

import admin_state
import alerts

logger = logging.getLogger(__name__)

CASCADE_PATH = 'haarcascade_frontalface_default.xml'
MODEL_PATH = 'trainer.yml'


def _load_recognizer():
    if not os.path.exists(MODEL_PATH):
        # ValueError so the view can show this text to the operator directly,
        # rather than the generic failure message.
        raise ValueError(
            "No trained model found. Train the model before starting detection.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
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


def run(frames, names, threshold):
    """Recognise faces in `frames` until the operator presses q.

    `names` maps an LBPH label to a username. A label that is not in the map is
    treated as unrecognised rather than logged under someone else's name.
    """
    recognizer = _load_recognizer()
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    font = cv2.FONT_HERSHEY_SIMPLEX

    alert_number = admin_state.read(admin_state.MOBILE_NO).strip()
    frames_to_log = settings.FACE_FRAMES_TO_LOG
    frames_to_alert = settings.FACE_FRAMES_TO_ALERT

    valid = 0
    invalid = 0

    try:
        for img in frames:
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                label, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                name = names.get(label)

                if confidence < threshold and name:
                    valid += 1
                    if valid >= frames_to_log:
                        _log_entry(name)
                        valid = 0
                        invalid = 0
                        caption = "Logged to system"
                        cv2.putText(img, caption, (x + 5, y - 5), font, 1,
                                    (255, 255, 255), 2)
                        cv2.imshow('camera', img)
                        cv2.waitKey(1)
                        time.sleep(3)
                        continue
                    caption = "Detected " + name
                else:
                    invalid += 1
                    if invalid >= frames_to_alert:
                        _send_alert(alert_number)
                        valid = 0
                        invalid = 0
                        caption = "Unrecognised face - system alerted"
                    else:
                        caption = "Detecting.."

                cv2.putText(img, caption, (x + 5, y - 5), font, 1,
                            (255, 255, 255), 2)

            cv2.imshow('camera', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # The frame source releases its own capture; this only owns the window.
        cv2.destroyAllWindows()
