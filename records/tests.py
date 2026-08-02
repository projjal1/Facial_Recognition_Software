"""Enrolment: uploads, the streaming capture, and who may reach them.

The camera is replaced with a finite generator throughout. That is the whole
point of expressing frame sources as generators - the capture paths become
testable without hardware, and the stream terminates instead of running until a
viewer disconnects.
"""

import io
import os
import shutil
import tempfile
from unittest import mock

import numpy as np
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image

import face_store

FRAME_COUNT = 6


def fake_frames(*args, **kwargs):
    for _ in range(FRAME_COUNT):
        yield np.zeros((64, 64, 3), dtype=np.uint8)


def png_upload(name='face.png', mode='RGBA'):
    """An in-memory PNG, so the upload path is exercised with a real image."""
    buffer = io.BytesIO()
    Image.new(mode, (32, 32), (128, 128, 128, 255)[:len(mode)]).save(buffer, 'PNG')
    buffer.seek(0)
    return SimpleUploadedFile(name, buffer.read(), content_type='image/png')


class EnrolmentTestCase(TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.override = override_settings(FACE_IMAGE_ROOT=self.root,
                                          FACE_ENROLMENT_FRAMES=3)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(shutil.rmtree, self.root, True)

        User.objects.create_user('s1', password='pw')
        self.client.login(username='s1', password='pw')


class AccessControlTests(TestCase):

    def test_anonymous_is_sent_to_the_sign_in_page(self):
        for name in ['reg-face', 'add-face', 'enrol-stream-local',
                     'enrol-stream-remote']:
            with self.subTest(name=name):
                response = self.client.get(reverse(name))
                self.assertEqual(response.status_code, 302)
                self.assertIn(reverse('login'), response.url)


class UploadTests(EnrolmentTestCase):

    def test_saves_an_uploaded_image(self):
        self.client.post(reverse('pic'), {'id_image': png_upload()})
        self.assertEqual(face_store.image_count('s1'), 1)

    def test_converts_transparency_rather_than_failing(self):
        # A PNG with an alpha channel cannot be written as JPEG directly.
        response = self.client.post(reverse('pic'), {'id_image': png_upload()})
        self.assertContains(response, 'uploaded')

    def test_numbers_uploads_after_the_existing_ones(self):
        self.client.post(reverse('pic'), {'id_image': png_upload()})
        self.client.post(reverse('pic'), {'id_image': png_upload()})
        names = sorted(os.listdir(face_store.folder_for('s1')))
        self.assertEqual(names, ['img1.jpg', 'img2.jpg'])

    def test_reports_a_file_that_is_not_an_image(self):
        bad = SimpleUploadedFile('notes.jpg', b'this is not an image',
                                 content_type='image/jpeg')
        response = self.client.post(reverse('pic'), {'id_image': bad})
        self.assertContains(response, 'could not be read')
        self.assertEqual(face_store.image_count('s1'), 0)

    def test_reports_a_missing_file(self):
        response = self.client.post(reverse('pic'), {})
        self.assertContains(response, 'No image was selected')


class LocalStreamTests(EnrolmentTestCase):

    @mock.patch('camera.local_frames', fake_frames)
    def test_streams_multipart_jpeg(self):
        response = self.client.get(reverse('enrol-stream-local'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('multipart/x-mixed-replace', response['Content-Type'])

        body = b''.join(response.streaming_content)
        self.assertEqual(body.count(b'--frame'), FRAME_COUNT)

    @mock.patch('camera.local_frames', fake_frames)
    def test_the_page_hosting_the_stream_renders(self):
        response = self.client.get(reverse('add-face'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse('enrol-stream-local'))

    def test_a_camera_that_will_not_open_is_reported_as_a_page(self):
        # The error has to surface before the response starts streaming,
        # otherwise there is no status code left to send.
        def refuses(*args, **kwargs):
            raise ValueError('Could not open the server webcam.')
            yield  # pragma: no cover - generator marker

        with mock.patch('camera.local_frames', refuses):
            response = self.client.get(reverse('enrol-stream-local'))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn('multipart', response['Content-Type'])
        self.assertContains(response, 'Could not open the server webcam')


class RemoteStreamTests(EnrolmentTestCase):

    def test_refuses_without_a_url_in_the_session(self):
        response = self.client.get(reverse('enrol-stream-remote'))
        self.assertContains(response, 'No camera URL')

    def test_a_refused_url_never_reaches_the_session(self):
        response = self.client.post(reverse('add-face-remote'),
                                    {'link': 'http://127.0.0.1/shot.jpg'})
        self.assertContains(response, 'loopback')
        self.assertNotIn('enrol_camera_url', self.client.session)

    @mock.patch('camera.remote_frames', fake_frames)
    def test_an_accepted_url_is_carried_to_the_stream(self):
        self.client.post(reverse('add-face-remote'),
                         {'link': 'http://192.168.1.50/shot.jpg'})
        self.assertEqual(self.client.session['enrol_camera_url'],
                         'http://192.168.1.50/shot.jpg')

        response = self.client.get(reverse('enrol-stream-remote'))
        self.assertIn('multipart/x-mixed-replace', response['Content-Type'])
        body = b''.join(response.streaming_content)
        self.assertEqual(body.count(b'--frame'), FRAME_COUNT)


class NonEnrollableAccountTests(TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.override = override_settings(FACE_IMAGE_ROOT=self.root)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_an_account_outside_the_naming_rule_is_told_why(self):
        # Superusers are created on the command line and bypass signup, so an
        # account that cannot own a face folder does reach these pages.
        User.objects.create_superuser('boss', 'boss@example.org', 'pw')
        self.client.login(username='boss', password='pw')
        response = self.client.get(reverse('reg-face'))
        self.assertContains(response, 'cannot enrol faces')
