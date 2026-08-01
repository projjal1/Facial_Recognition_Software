"""Frame sources for the two capture paths.

Enrolment and recognition each read from either the webcam attached to the
server or a remote camera serving JPEG snapshots over HTTP - four combinations
that previously carried four copies of the same setup code. Expressing each
source as a generator keeps the capture loops free of device handling, and the
`finally` guarantees the camera is released even when a loop exits early or
raises, which the original code did not.
"""

import logging

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

# Without a timeout a stalled camera hangs the request thread indefinitely.
REQUEST_TIMEOUT = 10


def local_frames(flip=True):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise ValueError(
            "Could not open the server webcam. Check that it is connected and "
            "not already in use by another program.")

    try:
        while True:
            ok, img = cam.read()
            if not ok:
                continue
            yield cv2.flip(img, 1) if flip else img
    finally:
        cam.release()


def remote_frames(url, flip=False):
    while True:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        buffer = np.frombuffer(response.content, dtype=np.uint8)
        # IMREAD_COLOR rather than IMREAD_UNCHANGED, so a greyscale or
        # alpha-channel snapshot still arrives as the 3-channel image the
        # colour conversion downstream expects.
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(
                "The camera URL did not return an image this program can read.")

        yield cv2.flip(img, 1) if flip else img
