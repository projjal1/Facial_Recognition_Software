"""MJPEG framing and the pre-flight pull.

`primed` is the more important of the two: once a streaming response has
started there is no status code left to send, so a setup failure has to surface
before any bytes go out or the viewer just sees a broken image.
"""

import numpy as np
from django.test import SimpleTestCase

import streaming


def frames(count, size=32):
    for _ in range(count):
        yield np.zeros((size, size, 3), dtype=np.uint8)


class MjpegTests(SimpleTestCase):

    def test_emits_one_part_per_frame(self):
        body = b''.join(streaming.mjpeg(frames(4)))
        self.assertEqual(body.count(b'--frame'), 4)

    def test_part_is_a_well_formed_multipart_chunk(self):
        part = next(streaming.mjpeg(frames(1)))
        self.assertTrue(part.startswith(b'--frame\r\n'))
        self.assertIn(b'Content-Type: image/jpeg\r\n', part)
        self.assertTrue(part.endswith(b'\r\n'))

    def test_payload_is_a_real_jpeg(self):
        part = next(streaming.mjpeg(frames(1)))
        payload = part.split(b'\r\n\r\n', 1)[1]
        self.assertTrue(payload.startswith(b'\xff\xd8'))  # JPEG start-of-image

    def test_content_length_matches_the_payload(self):
        part = next(streaming.mjpeg(frames(1)))
        declared = int(part.split(b'Content-Length: ')[1].split(b'\r\n')[0])
        payload = part.split(b'\r\n\r\n', 1)[1]
        self.assertEqual(declared, len(payload) - 2)  # minus the trailing CRLF

    def test_an_empty_source_produces_no_output(self):
        self.assertEqual(list(streaming.mjpeg(iter([]))), [])

    def test_content_type_declares_the_boundary(self):
        self.assertIn(streaming.BOUNDARY, streaming.CONTENT_TYPE)
        self.assertIn('multipart/x-mixed-replace', streaming.CONTENT_TYPE)


class PrimedTests(SimpleTestCase):

    def test_setup_failure_surfaces_to_the_caller(self):
        def explodes():
            raise ValueError('camera missing')
            yield  # pragma: no cover - generator marker

        with self.assertRaises(ValueError):
            streaming.primed(explodes())

    def test_empty_source_raises_stop_iteration(self):
        # The views catch this and report it rather than streaming nothing.
        with self.assertRaises(StopIteration):
            streaming.primed(iter([]))

    def test_no_frame_is_lost_to_the_pre_flight_pull(self):
        self.assertEqual(list(streaming.primed(iter([1, 2, 3]))), [1, 2, 3])

    def test_the_source_is_not_drained_eagerly(self):
        pulled = []

        def counted():
            for i in range(3):
                pulled.append(i)
                yield i

        primed = streaming.primed(counted())
        self.assertEqual(pulled, [0])  # only the first frame so far
        list(primed)
        self.assertEqual(pulled, [0, 1, 2])
