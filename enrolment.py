"""Shared enrolment capture: save cropped faces until the target count is met.

start.py and remote_start.py previously held separate copies of this, which is
why only one of them cleaned up after itself and only one stopped cleanly at the
frame limit.

Like the recognition loop, this yields annotated frames instead of drawing them,
so the same code can feed a video stream in the browser. It finishes on its own
once enough faces are captured, which is what ends the stream and tells the
viewer the run is over.
"""

import logging
import os

import cv2
from django.conf import settings

logger = logging.getLogger(__name__)

CASCADE_PATH = 'haarcascade_frontalface_default.xml'


def capture(source, folder, existing):
    """Yield annotated frames while saving face crops into `folder`.

    Images are numbered after `existing` so a second run adds to the set rather
    than overwriting it. Stops once settings.FACE_ENROLMENT_FRAMES faces have
    been saved.
    """
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    font = cv2.FONT_HERSHEY_SIMPLEX
    target = settings.FACE_ENROLMENT_FRAMES
    saved = 0

    os.makedirs(folder, exist_ok=True)

    for img in source:
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

        cv2.putText(img, "Captured %d of %d" % (saved, target), (10, 30),
                    font, 1, (255, 255, 255), 2)
        yield img

        if saved >= target:
            logger.info("Saved %d enrolment images to %s.", saved, folder)
            return
