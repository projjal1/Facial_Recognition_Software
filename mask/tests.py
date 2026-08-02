"""The experimental mask detector.

As with the emotion app, these assert the plumbing rather than the predictions:
access control, and that the page streams instead of opening a window on the
server.
"""

from unittest import mock

import numpy as np
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

FRAME_COUNT = 4


def fake_frames(*args, **kwargs):
    for _ in range(FRAME_COUNT):
        yield np.zeros((64, 64, 3), dtype=np.uint8)


class AccessControlTests(TestCase):

    def test_anonymous_is_sent_to_the_sign_in_page(self):
        for name in ['face_mask', 'mask-stream']:
            with self.subTest(name=name):
                response = self.client.get(reverse(name))
                self.assertEqual(response.status_code, 302)
                self.assertIn(reverse('login'), response.url)


class StreamTests(TestCase):

    def setUp(self):
        User.objects.create_user('s1', password='pw')
        self.client.login(username='s1', password='pw')

    def test_the_page_points_at_the_stream(self):
        response = self.client.get(reverse('face_mask'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse('mask-stream'))

    @mock.patch('camera.local_frames', fake_frames)
    def test_streams_multipart_jpeg(self):
        response = self.client.get(reverse('mask-stream'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('multipart/x-mixed-replace', response['Content-Type'])

        body = b''.join(response.streaming_content)
        self.assertEqual(body.count(b'--frame'), FRAME_COUNT)

    def test_a_camera_that_will_not_open_is_reported_as_an_image(self):
        # The stream is loaded by an <img>, so an HTML error page would only
        # ever show up as a broken image.
        def refuses(*args, **kwargs):
            raise ValueError('Could not open the server webcam.')
            yield  # pragma: no cover - generator marker

        with mock.patch('camera.local_frames', refuses):
            response = self.client.get(reverse('mask-stream'))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'image/jpeg')
        self.assertTrue(response.content.startswith(b'\xff\xd8'))


class ModelTests(TestCase):

    def test_the_bundled_keras_2_model_still_runs(self):
        # Depends on tf-keras being selected in place of Keras 3. Loading and
        # predicting are different things, so this asserts a prediction.
        from mask.resources import webcam

        scores = webcam.model.predict(
            np.zeros((1, 224, 224, 3), dtype='float32'), verbose=0)
        self.assertEqual(scores.shape, (1, 3))

    def test_the_face_detector_runs_over_a_frame(self):
        from mask.resources import webcam

        locs, preds = webcam.detect_and_predict_mask(
            np.zeros((300, 300, 3), dtype='uint8'), webcam.faceNet, webcam.model)
        self.assertIsInstance(locs, list)
