"""Flat-file runtime state under admin_files/.

The behaviour worth pinning down is what happens when a file is absent or
empty, because a fresh checkout has neither the entry log nor the alert number,
and every reader has to treat that as ordinary rather than as an error.
"""

import os
import shutil
import tempfile

from django.test import SimpleTestCase, override_settings

import admin_state


class AdminStateTests(SimpleTestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        # admin_state resolves everything under BASE_DIR, so pointing that at a
        # temporary directory keeps the real admin_files/ untouched.
        self.override = override_settings(BASE_DIR=self.root)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_reading_a_missing_file_gives_empty_string(self):
        self.assertEqual(admin_state.read(admin_state.LOGS), '')

    def test_write_creates_the_directory(self):
        # Nothing has created admin_files/ in the temporary root yet.
        admin_state.write(admin_state.MOBILE_NO, '5550000')
        self.assertEqual(admin_state.read(admin_state.MOBILE_NO), '5550000')

    def test_write_replaces_and_append_adds(self):
        admin_state.write(admin_state.LOGS, 'first')
        admin_state.write(admin_state.LOGS, 'second')
        self.assertEqual(admin_state.read(admin_state.LOGS), 'second')

        admin_state.append(admin_state.LOGS, '\nthird')
        self.assertEqual(admin_state.read(admin_state.LOGS), 'second\nthird')

    def test_read_int_defaults_when_absent_or_empty(self):
        self.assertEqual(admin_state.read_int(admin_state.TRAINED), 0)

        admin_state.write(admin_state.TRAINED, '')
        self.assertEqual(admin_state.read_int(admin_state.TRAINED), 0)

        admin_state.write(admin_state.TRAINED, '   \n')
        self.assertEqual(admin_state.read_int(admin_state.TRAINED), 0)

    def test_read_int_parses_a_count(self):
        admin_state.write(admin_state.TRAINED, '407')
        self.assertEqual(admin_state.read_int(admin_state.TRAINED), 407)

    def test_read_int_uses_the_last_line(self):
        # recog.py rewrites the file, but a stray earlier line should not win.
        admin_state.write(admin_state.TRAINED, '12\n34')
        self.assertEqual(admin_state.read_int(admin_state.TRAINED), 34)

    def test_read_int_falls_back_when_the_contents_are_not_a_number(self):
        admin_state.write(admin_state.TRAINED, 'not a number')
        self.assertEqual(admin_state.read_int(admin_state.TRAINED), 0)
        self.assertEqual(admin_state.read_int(admin_state.TRAINED, default=7), 7)

    def test_path_stays_inside_admin_files(self):
        self.assertEqual(
            admin_state.path(admin_state.LINK),
            os.path.join(self.root, 'admin_files', 'link.txt'))
