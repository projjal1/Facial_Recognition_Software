"""Shared enrolment capture: save normalised face crops until the target is met.

What gets written is the same fixed-size, contrast-equalised crop that
recognition will later match against, produced by the same code in faces.py.
Storing raw frames and normalising later is how the two halves drift apart.

Blurred crops are rejected rather than counted. Fifteen smeared images teach the
model a smeared version of someone, and cost more accuracy than they add.
"""

import logging
import os

import cv2
from django.conf import settings

import faces

logger = logging.getLogger(__name__)


def capture(source, folder, existing):
    """Yield annotated frames while saving normalised crops into `folder`.

    Images are numbered after `existing` so a second run adds to the set rather
    than overwriting it. Stops once settings.FACE_ENROLMENT_FRAMES are saved.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    target = settings.FACE_ENROLMENT_FRAMES
    saved = 0
    skipped = 0

    os.makedirs(folder, exist_ok=True)

    for img in source:
        if img is None:
            continue

        for box, crop in faces.crops(img):
            x, y, w, h = box

            if not faces.is_sharp(crop):
                skipped += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 140, 255), 2)
                cv2.putText(img, "Too blurred - hold still", (x + 5, y - 5),
                            font, 0.6, (255, 255, 255), 2)
                continue

            saved += 1
            cv2.imwrite(os.path.join(folder, "%d.jpg" % (existing + saved)), crop)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if saved >= target:
                break

        cv2.putText(img, "Captured %d of %d" % (saved, target), (10, 30),
                    font, 0.8, (255, 255, 255), 2)
        yield img

        if saved >= target:
            logger.info("Saved %d crops to %s (%d rejected as blurred).",
                        saved, folder, skipped)
            return
