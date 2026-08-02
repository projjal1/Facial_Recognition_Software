"""Shared enrolment capture: save normalised face crops until the target is met.

What gets written is the same fixed-size, contrast-equalised crop that
recognition will later match against, produced by the same code in faces.py.
Storing raw frames and normalising later is how the two halves drift apart.

Two things are refused rather than saved. Blurred crops, because fifteen
smeared images teach the model a smeared version of someone. And a face that
already belongs to somebody else, because enrolling one person under two names
puts half their images in each class and leaves the recogniser unable to
separate them - a failure that shows up later as poor accuracy with no obvious
cause.
"""

import logging
import os
from collections import Counter

import cv2
from django.conf import settings

import faces
import recog

logger = logging.getLogger(__name__)


class _IdentityCheck:
    """Decides whether the face being enrolled is already on file.

    Judged over several crops rather than one, because a single frame is a
    coin toss near the threshold. Nothing is written to disk until this
    settles, so a run that turns out to be a duplicate leaves no trace.
    """

    def __init__(self, label):
        self.label = label
        self.model = None if label is None else recog.load_model()
        # With no model, or no label to compare against, there is nothing to do.
        self.settled = self.model is None
        self.duplicate = None
        self._votes = Counter()
        self._seen = 0

    def consider(self, crop):
        predicted, distance = self.model.predict(crop)
        if predicted != self.label and distance < settings.FACE_DUPLICATE_THRESHOLD:
            self._votes[predicted] += 1

        self._seen += 1
        if self._seen < settings.FACE_DUPLICATE_CHECK_FRAMES:
            return

        self.settled = True
        if self._votes:
            match, count = self._votes.most_common(1)[0]
            # A majority, so one stray frame cannot block a genuine enrolment.
            if count * 2 > self._seen:
                self.duplicate = 's%d' % match


def capture(source, folder, existing, label=None):
    """Yield annotated frames while saving normalised crops into `folder`.

    Images are numbered after `existing` so a second run adds to the set rather
    than overwriting it. Stops once settings.FACE_ENROLMENT_FRAMES are saved,
    or immediately if the face turns out to be someone already enrolled.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    target = settings.FACE_ENROLMENT_FRAMES
    check = _IdentityCheck(label)

    saved = 0
    skipped = 0
    held = []

    os.makedirs(folder, exist_ok=True)

    def write(crop):
        nonlocal saved
        saved += 1
        cv2.imwrite(os.path.join(folder, "%d.jpg" % (existing + saved)), crop)

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

            if not check.settled:
                check.consider(crop)
                held.append(crop)

                if not check.settled:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    cv2.putText(img, "Checking...", (x + 5, y - 5),
                                font, 0.6, (255, 255, 255), 2)
                    continue

                if check.duplicate:
                    logger.warning(
                        "Enrolment stopped: this face is already enrolled as %s.",
                        check.duplicate)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img,
                                "Already enrolled as %s" % check.duplicate,
                                (x + 5, y - 5), font, 0.6, (255, 255, 255), 2)
                    yield img
                    return

                # Cleared: everything held back is genuine, so write it now.
                for pending in held:
                    write(pending)
                held.clear()
            else:
                write(crop)

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
