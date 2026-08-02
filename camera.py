"""Frame sources for the two capture paths.

Enrolment and recognition each read from either the webcam attached to the
server or a remote camera serving JPEG snapshots over HTTP - four combinations
that previously carried four copies of the same setup code. Expressing each
source as a generator keeps the capture loops free of device handling, and the
`finally` guarantees the camera is released even when a consumer stops early.
"""

import logging
import threading

import cv2
import numpy as np
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# Without a timeout a stalled camera hangs the worker thread indefinitely.
REQUEST_TIMEOUT = 10

# A camera that has stopped returning frames should end the stream rather than
# spin reading failures forever.
MAX_READ_FAILURES = 60


class _Device:
    """Serialises access to the one webcam, with the newest request winning.

    Navigating from one capture page to another is the case that matters. The
    browser does not tell the server it has gone away; the old response only
    ends once a write to its socket fails, and that waits for the send buffer
    to fill - easily longer than someone is willing to stare at a blank frame.
    Refusing the new page until then means switching between the emotion, mask
    and enrolment pages appears broken.

    So a new claim asks whoever holds the camera to stop, then queues for it.
    The incumbent notices between frames and releases on its way out.
    """

    def __init__(self):
        self._owner = threading.Lock()
        self._state = threading.Lock()
        self._stop = None

    def claim(self, timeout):
        """Take the camera, returning the Event that asks us to hand it back."""
        with self._state:
            if self._stop is not None:
                logger.info("Asking the current capture to release the camera.")
                self._stop.set()

        if not self._owner.acquire(timeout=timeout):
            raise ValueError(
                "The camera is still in use by another capture and did not "
                "come free. Close the other page and try again.")

        with self._state:
            self._stop = threading.Event()
            return self._stop

    def relinquish(self, token):
        with self._state:
            if self._stop is token:
                self._stop = None
        self._owner.release()

    def request_stop(self):
        """Ask the current holder to finish. True if there was one."""
        with self._state:
            if self._stop is None:
                return False
            self._stop.set()
            return True


_device = _Device()


def release_current():
    """Stop whatever capture is running, if any.

    Leaving a capture page is not enough on its own: the stream ends only when
    a write to its socket fails, and until then the detector keeps running a
    model over every frame for nobody's benefit. Giving the pages a way to say
    "I am done" turns that into an immediate stop.
    """
    stopped = _device.request_stop()
    if stopped:
        logger.info("Asked the running capture to stop.")
    return stopped


def local_frames(flip=True):
    stop = _device.claim(settings.FACE_CAMERA_HANDOVER_SECONDS)

    cam = None
    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise ValueError(
                "Could not open the server webcam. Check that it is connected "
                "and not already in use by another program.")

        failures = 0
        # Checked between frames, so a page that takes the camera over waits
        # roughly one frame rather than for the old connection to time out.
        while not stop.is_set():
            ok, img = cam.read()
            if not ok:
                failures += 1
                if failures > MAX_READ_FAILURES:
                    raise ValueError(
                        "The webcam stopped returning frames. Check that it is "
                        "still connected.")
                continue

            failures = 0
            yield cv2.flip(img, 1) if flip else img
    finally:
        if cam is not None:
            cam.release()
        _device.relinquish(stop)
        logger.info("Released the local webcam.")


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
