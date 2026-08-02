"""The superuser control panel: label mapping, training status, access.

The label mapping is the one to get right. Recognition looks a person up by the
number in their folder name, and the previous implementation built a list in
database order and indexed into it - so deleting an account shifted every name,
and a label past the end of the list raised mid-capture.
"""

import os
import shutil
import tempfile

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse

import admin_state
from feeds.views import _label_names, _not_ready, _training_status


class LabelNameTests(TestCase):

    def test_maps_each_label_to_its_username(self):
        for name in ['s1', 's2']:
            User.objects.create_user(name, password='pw')
        self.assertEqual(_label_names(), {1: 's1', 2: 's2'})

    def test_survives_gaps_in_the_numbering(self):
        # A deleted account used to shift every later name by one.
        for name in ['s1', 's3', 's12']:
            User.objects.create_user(name, password='pw')
        self.assertEqual(_label_names(), {1: 's1', 3: 's3', 12: 's12'})

    def test_ignores_accounts_that_own_no_face_folder(self):
        User.objects.create_user('s1', password='pw')
        User.objects.create_superuser('boss', 'boss@example.org', 'pw')
        User.objects.create_user('sam', password='pw')
        self.assertEqual(_label_names(), {1: 's1'})

    def test_is_empty_when_nobody_has_signed_up(self):
        self.assertEqual(_label_names(), {})


class TrainingStatusTests(TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.store = os.path.join(self.root, 'face-files')
        self.override = override_settings(BASE_DIR=self.root,
                                          FACE_IMAGE_ROOT=self.store)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(shutil.rmtree, self.root, True)

    def _enrol(self, username, images):
        folder = os.path.join(self.store, username)
        os.makedirs(folder, exist_ok=True)
        for i in range(images):
            open(os.path.join(folder, '%d.jpg' % i), 'w').close()

    def test_nothing_enrolled_and_nothing_trained_counts_as_up_to_date(self):
        self.assertIn('previously trained', _training_status())

    def test_reports_the_shortfall(self):
        self._enrol('s1', 5)
        admin_state.write(admin_state.TRAINED, '2')
        self.assertIn('3 pending', _training_status())

    def test_reports_up_to_date_when_the_counts_agree(self):
        self._enrol('s1', 4)
        admin_state.write(admin_state.TRAINED, '4')
        self.assertIn('previously trained', _training_status())


class NotReadyTests(TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.override = override_settings(FACE_IMAGE_ROOT=self.root)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_complains_when_nobody_is_enrolled(self):
        self.assertIn('No enrolled accounts', _not_ready())

    def test_complains_when_there_is_no_trained_model(self):
        # Guarding here rather than only inside the stream is what lets the
        # operator see why, instead of a broken image with no explanation.
        User.objects.create_user('s1', password='pw')
        with override_settings():
            import recognition
            original = recognition.MODEL_PATH
            recognition.MODEL_PATH = os.path.join(self.root, 'absent.yml')
            self.addCleanup(setattr, recognition, 'MODEL_PATH', original)
            self.assertIn('No trained model', _not_ready())


class AccessControlTests(TestCase):

    def setUp(self):
        User.objects.create_user('s1', password='pw')
        User.objects.create_superuser('boss', 'boss@example.org', 'pw')

    def test_anonymous_is_sent_to_the_sign_in_page(self):
        for name in ['feed-detect', 'start', 'detect-stream', 'end']:
            with self.subTest(name=name):
                response = self.client.get(reverse(name))
                self.assertEqual(response.status_code, 302)
                self.assertIn(reverse('login'), response.url)

    def test_a_signed_in_user_who_is_not_a_superuser_is_refused(self):
        self.client.login(username='s1', password='pw')
        for name in ['feed-detect', 'start', 'detect-stream', 'train']:
            with self.subTest(name=name):
                self.assertEqual(self.client.get(reverse(name)).status_code, 302)

    def test_a_superuser_reaches_the_panel(self):
        self.client.login(username='boss', password='pw')
        self.assertEqual(self.client.get(reverse('feed-detect')).status_code, 200)


class RemoteSourceTests(TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.override = override_settings(BASE_DIR=self.root)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(shutil.rmtree, self.root, True)

        User.objects.create_superuser('boss', 'boss@example.org', 'pw')
        self.client.login(username='boss', password='pw')

    def test_a_refused_camera_url_is_not_stored(self):
        response = self.client.post(reverse('with_url'),
                                    {'url': 'http://127.0.0.1/shot.jpg'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(admin_state.read(admin_state.LINK), '')

    def test_an_acceptable_camera_url_is_stored(self):
        self.client.post(reverse('with_url'), {'url': 'http://192.168.1.50/shot.jpg'})
        self.assertEqual(admin_state.read(admin_state.LINK),
                         'http://192.168.1.50/shot.jpg')

    def test_choosing_the_local_camera_clears_the_url(self):
        admin_state.write(admin_state.LINK, 'http://192.168.1.50/shot.jpg')
        self.client.post(reverse('without_url'))
        self.assertEqual(admin_state.read(admin_state.LINK), '')
