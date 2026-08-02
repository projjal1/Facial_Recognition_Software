"""The experimental emotion detector.

These assert the plumbing - access control, and that the page streams rather
than opening a window on the server. The model's predictions are not asserted;
they are indicative at best, and pinning them would only make the test brittle.
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
        # The page opens the camera on the server, so it needs an account
        # behind it even though the feature is not admin-only.
        for name in ['emotion', 'emotion-stream']:
            with self.subTest(name=name):
                response = self.client.get(reverse(name))
                self.assertEqual(response.status_code, 302)
                self.assertIn(reverse('login'), response.url)


class StreamTests(TestCase):

    def setUp(self):
        User.objects.create_user('s1', password='pw')
        self.client.login(username='s1', password='pw')

    def test_the_page_points_at_the_stream(self):
        response = self.client.get(reverse('emotion'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse('emotion-stream'))

    @mock.patch('camera.local_frames', fake_frames)
    def test_streams_multipart_jpeg(self):
        response = self.client.get(reverse('emotion-stream'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('multipart/x-mixed-replace', response['Content-Type'])

        body = b''.join(response.streaming_content)
        self.assertEqual(body.count(b'--frame'), FRAME_COUNT)

    def test_a_camera_that_will_not_open_is_reported_as_a_page(self):
        def refuses(*args, **kwargs):
            raise ValueError('Could not open the server webcam.')
            yield  # pragma: no cover - generator marker

        with mock.patch('camera.local_frames', refuses):
            response = self.client.get(reverse('emotion-stream'))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn('multipart', response['Content-Type'])
        self.assertContains(response, 'Could not open the server webcam')


class ModelTests(TestCase):

    def test_the_bundled_keras_2_model_still_runs(self):
        # TensorFlow 2.16 switched tf.keras to Keras 3, which cannot load these
        # weights; the app depends on tf-keras being selected instead. Loading
        # and predicting are different things, so this asserts a prediction.
        from emotion.resources import cam

        prediction = cam.model.predict_emotion(
            np.zeros((1, 48, 48, 1), dtype='float32'))
        self.assertIn(prediction, cam.model.EMOTIONS_LIST)
