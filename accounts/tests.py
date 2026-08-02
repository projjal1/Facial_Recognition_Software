"""Signup validation, session handling, and who can reach the admin pages.

The access-control tests exist because the navigation bar hides links by role
while the URLs themselves answered to anyone until decorators were added - so
what is worth asserting is the response to a request that never went through
the navigation at all.
"""

import os
import shutil
import tempfile

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse

import admin_state


class SignupTests(TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.override = override_settings(FACE_IMAGE_ROOT=self.root)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(shutil.rmtree, self.root, True)

    def post(self, **overrides):
        data = {'username': 's1', 'pass1': 'pw', 'pass2': 'pw',
                'first': 'Ada', 'last': 'Lovelace'}
        data.update(overrides)
        return self.client.post(reverse('signup'), data)

    def test_creates_an_account_and_its_image_folder(self):
        self.post()
        self.assertTrue(User.objects.filter(username='s1').exists())
        self.assertTrue(os.path.isdir(os.path.join(self.root, 's1')))

    def test_signs_the_new_user_in(self):
        response = self.post()
        self.assertRedirects(response, reverse('home'))
        self.assertIn('_auth_user_id', self.client.session)

    def test_rejects_a_username_outside_s_number(self):
        # Such an account could never be trained, so it used to be created and
        # then silently never recognised.
        for username in ['ada', 'admin', 'static', 's', 'S1']:
            with self.subTest(username=username):
                self.post(username=username)
                self.assertFalse(User.objects.filter(username=username).exists())

    def test_rejects_a_username_that_would_reach_the_shell_or_filesystem(self):
        for username in ['s1; rm -rf /', '../etc', 's1/../s2']:
            with self.subTest(username=username):
                self.post(username=username)
                self.assertFalse(User.objects.filter(username=username).exists())

    def test_rejects_mismatched_passwords(self):
        self.post(pass2='different')
        self.assertFalse(User.objects.filter(username='s1').exists())

    def test_rejects_missing_details(self):
        self.post(first='')
        self.assertFalse(User.objects.filter(username='s1').exists())

    def test_rejects_a_duplicate_username(self):
        User.objects.create_user('s1', password='pw')
        self.post(first='Someone', last='Else')
        self.assertEqual(User.objects.filter(username='s1').count(), 1)


class LogoutTests(TestCase):

    def setUp(self):
        User.objects.create_user('s1', password='pw')
        self.client.login(username='s1', password='pw')

    def test_post_ends_the_session(self):
        self.client.post(reverse('logout'))
        self.assertNotIn('_auth_user_id', self.client.session)

    def test_get_redirects_without_ending_the_session(self):
        # A GET used to fall off the end of the view and return None, which
        # Django raises on. Only POST should log out, so that a link prefetch
        # cannot end someone's session.
        response = self.client.get(reverse('logout'))
        self.assertRedirects(response, reverse('home'))
        self.assertIn('_auth_user_id', self.client.session)


class AccessControlTests(TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.override = override_settings(BASE_DIR=self.root,
                                          FACE_IMAGE_ROOT=self.root)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(shutil.rmtree, self.root, True)

        User.objects.create_user('s1', password='pw')
        User.objects.create_superuser('boss', 'boss@example.org', 'pw')

    def test_anonymous_is_sent_to_the_sign_in_page(self):
        for name in ['prof', 'logs', 'about']:
            with self.subTest(name=name):
                response = self.client.get(reverse(name))
                self.assertEqual(response.status_code, 302)
                self.assertIn(reverse('login'), response.url)

    def test_a_signed_in_user_cannot_reach_the_admin_pages(self):
        self.client.login(username='s1', password='pw')
        for name in ['prof', 'logs']:
            with self.subTest(name=name):
                self.assertEqual(self.client.get(reverse(name)).status_code, 302)

    def test_a_superuser_can(self):
        self.client.login(username='boss', password='pw')
        for name in ['prof', 'logs']:
            with self.subTest(name=name):
                self.assertEqual(self.client.get(reverse(name)).status_code, 200)

    def test_a_signed_in_user_reaches_their_own_page(self):
        self.client.login(username='s1', password='pw')
        self.assertEqual(self.client.get(reverse('about')).status_code, 200)

    def test_the_home_page_stays_public(self):
        self.assertEqual(self.client.get(reverse('home')).status_code, 200)


class AdminPageTests(TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.override = override_settings(BASE_DIR=self.root)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(shutil.rmtree, self.root, True)

        User.objects.create_superuser('boss', 'boss@example.org', 'pw')
        self.client.login(username='boss', password='pw')

    def test_alert_number_round_trips(self):
        self.client.post(reverse('prof'), {'mobile': '5550000'})
        self.assertContains(self.client.get(reverse('prof')), '5550000')

    def test_pages_work_before_any_state_files_exist(self):
        # A fresh checkout has no admin_files/ at all; that is not an error.
        self.assertEqual(self.client.get(reverse('logs')).status_code, 200)
        self.assertEqual(self.client.get(reverse('prof')).status_code, 200)

    def test_posting_to_logs_clears_them(self):
        admin_state.write(admin_state.LOGS, '\n s1 logged at some point')
        self.client.post(reverse('logs'))
        self.assertEqual(admin_state.read(admin_state.LOGS), '')
