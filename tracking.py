"""Associating detections across frames.

Recognition has to know that the face in this frame is the same person as the
face in the last one. Without that there is a single counter shared by everyone
in view, so sixty confident frames of a crowd produce one log entry naming
whoever happened to be in the final frame.

Centroid distance is enough here. The camera is fixed and faces move slowly
relative to the frame rate, so the nearest recent track is almost always the
right one. A Kalman or KCF tracker would only earn its keep under occlusion or
fast motion, neither of which is what a doorway camera sees.

Votes accumulate per track, and a decision is made on the most-voted label
rather than on a run of consecutive agreeing frames - so a single dropped or
blurred frame no longer discards the evidence gathered so far.
"""

import itertools
from collections import Counter

from django.conf import settings


def _centre(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class Track:
    """One face followed across frames, with the votes cast for its identity."""

    def __init__(self, track_id, box):
        self.id = track_id
        self.box = box
        self.votes = Counter()
        self.misses = 0
        # Per track, so logging one person does not mute another.
        self.cooldown_until = 0.0

    @property
    def centre(self):
        return _centre(self.box)

    def record(self, name):
        """Note one frame's opinion. `None` means the face was not recognised."""
        self.votes[name] += 1

    def leader(self):
        """(name, count) for the most-voted identity, or (None, 0) if no votes."""
        if not self.votes:
            return None, 0
        name, count = self.votes.most_common(1)[0]
        return name, count

    def reset(self):
        self.votes.clear()


class Tracker:

    def __init__(self, max_distance=None, max_misses=None):
        self.max_distance = (settings.FACE_TRACK_MAX_DISTANCE
                             if max_distance is None else max_distance)
        self.max_misses = (settings.FACE_TRACK_MAX_MISSES
                           if max_misses is None else max_misses)
        self._tracks = []
        self._ids = itertools.count(1)

    @property
    def tracks(self):
        return list(self._tracks)

    def update(self, boxes):
        """Match `boxes` to existing tracks; return [(track, box)] in box order.

        Boxes are taken largest-first by the detector, so when two faces are
        equally close to one track the nearer-to-camera one wins it, which is
        the better guess at a doorway.
        """
        unmatched = list(self._tracks)
        paired = []

        for box in boxes:
            centre = _centre(box)
            best, best_distance = None, None

            for track in unmatched:
                distance = _distance(centre, track.centre)
                if distance > self.max_distance:
                    continue
                if best_distance is None or distance < best_distance:
                    best, best_distance = track, distance

            if best is None:
                best = Track(next(self._ids), box)
                self._tracks.append(best)
            else:
                unmatched.remove(best)

            best.box = box
            best.misses = 0
            paired.append((best, box))

        for track in unmatched:
            track.misses += 1

        # Drop tracks that have been gone long enough to be someone else by now.
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]

        return paired
