"""Shared enrolment capture: save cropped faces until the target count is met.

start.py and remote_start.py previously held separate copies of this, which is
why only one of them cleaned up its OpenCV window and only one of them stopped
cleanly at the frame limit.
"""

import logging
import os

import cv2
from django.conf import settings

logger = logging.getLogger(__name__)

CASCADE_PATH = 'haarcascade_frontalface_default.xml'


def capture(frames, folder, existing):
    """Save face crops from `frames` into `folder`, numbered after `existing`.

    Stops once settings.FACE_ENROLMENT_FRAMES images are saved, or when the
    operator presses q. Returns how many were written.
    """
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    target = settings.FACE_ENROLMENT_FRAMES
    saved = 0

    os.makedirs(folder, exist_ok=True)

    try:
        for img in frames:
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 3)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                saved += 1
                cv2.imwrite(
                    os.path.join(folder, "%d.jpg" % (existing + saved)),
                    gray[y:y + h, x:x + w])
                if saved >= target:
                    break

            cv2.imshow('image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if saved >= target:
                break
    finally:
        cv2.destroyAllWindows()

    logger.info("Saved %d enrolment images to %s.", saved, folder)
    return saved
