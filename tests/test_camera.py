"""Handing the one webcam between capture pages.

Navigating from the emotion page to the mask page is the case this exists for.
The browser gives the server no warning that it has gone, and the old response
only ends once a write to its socket fails - so the new page has to be able to
take the camera rather than wait for that.
"""

import threading
from unittest import mock

import numpy as np
from django.test import SimpleTestCase, override_settings

import camera


class FakeCam:
    opened = []

    def __init__(self, *args):
        self.released = False
        FakeCam.opened.append(self)

    def isOpened(self):
        return True

    def read(self):
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@override_settings(FACE_CAMERA_HANDOVER_SECONDS=5)
class HandoverTests(SimpleTestCase):

    def setUp(self):
        FakeCam.opened = []
        # A fresh device per test, so one test's holder cannot leak into another.
        original = camera._device
        camera._device = camera._Device()
        self.addCleanup(setattr, camera, '_device', original)

        patcher = mock.patch.object(
            camera, 'cv2',
            mock.Mock(VideoCapture=FakeCam, flip=lambda img, code: img))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_stream_releases_the_camera_when_closed(self):
        stream = camera.local_frames()
        next(stream)
        stream.close()
        self.assertTrue(FakeCam.opened[0].released)

    def test_a_second_stream_takes_over_from_the_first(self):
        first = camera.local_frames()
        next(first)

        second = camera.local_frames()
        # Claiming asks the incumbent to stop; it notices between frames.
        started = threading.Thread(target=lambda: next(second))
        started.start()

        # Draining the first lets it see the stop request and let go.
        list(first)
        started.join(timeout=5)

        self.assertFalse(started.is_alive())
        self.assertTrue(FakeCam.opened[0].released)
        second.close()

    def test_the_displaced_stream_ends_rather_than_running_on(self):
        first = camera.local_frames()
        next(first)

        camera._device._stop.set()  # what a new claim does

        # The loop checks between frames, so it finishes instead of yielding
        # for ever.
        self.assertEqual(list(first), [])

    def test_the_camera_is_released_after_an_error_mid_stream(self):
        class Broken(FakeCam):
            def read(self):
                raise RuntimeError('device fell over')

        camera.cv2.VideoCapture = Broken
        stream = camera.local_frames()
        with self.assertRaises(RuntimeError):
            next(stream)

        # The lock must be free, or every later capture is refused.
        camera.cv2.VideoCapture = FakeCam
        recovered = camera.local_frames()
        next(recovered)
        recovered.close()

    @override_settings(FACE_CAMERA_HANDOVER_SECONDS=0.2)
    def test_a_holder_that_never_lets_go_times_out_with_a_message(self):
        first = camera.local_frames()
        next(first)

        with self.assertRaises(ValueError) as caught:
            next(camera.local_frames())
        self.assertIn('did not come free', str(caught.exception))

        first.close()

    def test_read_failures_do_not_spin_for_ever(self):
        class NeverReady(FakeCam):
            def read(self):
                return False, None

        camera.cv2.VideoCapture = NeverReady
        with self.assertRaises(ValueError):
            next(camera.local_frames())
