"""Where enrolled face images live.

Two things are being protected here. Usernames become path components, so a
name that could escape the store must be refused; and labels are derived from
folder names, so the mapping has to survive gaps in the numbering.
"""

import os
import shutil
import tempfile

from django.test import SimpleTestCase, override_settings

import face_store


class FaceStoreTests(SimpleTestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.store = os.path.join(self.root, 'face-files')
        self.override = override_settings(FACE_IMAGE_ROOT=self.store)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(shutil.rmtree, self.root, True)

    def _enrol(self, username, images=0):
        folder = face_store.folder_for(username, create=True)
        for i in range(images):
            open(os.path.join(folder, '%d.jpg' % i), 'w').close()
        return folder

    def test_folder_is_created_under_the_store(self):
        folder = face_store.folder_for('s1', create=True)
        self.assertEqual(folder, os.path.join(self.store, 's1'))
        self.assertTrue(os.path.isdir(folder))

    def test_folder_for_does_not_create_unless_asked(self):
        face_store.folder_for('s1')
        self.assertFalse(os.path.exists(os.path.join(self.store, 's1')))

    def test_rejects_usernames_that_are_not_s_number(self):
        for username in ['admin', 'projjal', 'static', 'staticfiles', 's', 'S1', 's1a', '']:
            with self.assertRaises(ValueError, msg=username):
                face_store.folder_for(username)

    def test_rejects_names_that_would_escape_the_store(self):
        # The pattern is what makes the value safe to use as a path component.
        for username in ['../etc', 's1/../..', 's1; rm -rf /', 'C:\\windows']:
            with self.assertRaises(ValueError, msg=username):
                face_store.folder_for(username)

    def test_no_store_yet_is_not_an_error(self):
        self.assertEqual(list(face_store.enrolled_folders()), [])
        self.assertEqual(face_store.total_images(), 0)
        self.assertEqual(face_store.image_count('s1'), 0)

    def test_image_count_tolerates_a_missing_folder_and_a_bad_name(self):
        self.assertEqual(face_store.image_count('s9'), 0)
        self.assertEqual(face_store.image_count('admin'), 0)

    def test_enrolled_folders_keys_by_label_and_allows_gaps(self):
        self._enrol('s1')
        self._enrol('s3')
        self._enrol('s12')
        self.assertEqual([label for label, _ in face_store.enrolled_folders()],
                         [1, 3, 12])

    def test_enrolled_folders_sorts_numerically_not_alphabetically(self):
        # s12 would come before s3 under a plain string sort.
        self._enrol('s3')
        self._enrol('s12')
        labels = [label for label, _ in face_store.enrolled_folders()]
        self.assertEqual(labels, sorted(labels))

    def test_enrolled_folders_ignores_anything_else_in_the_store(self):
        self._enrol('s1')
        os.makedirs(os.path.join(self.store, 'notes'))
        open(os.path.join(self.store, 's2.txt'), 'w').close()
        self.assertEqual([label for label, _ in face_store.enrolled_folders()], [1])

    def test_counts_images(self):
        self._enrol('s1', images=2)
        self._enrol('s3', images=5)
        self.assertEqual(face_store.image_count('s1'), 2)
        self.assertEqual(face_store.total_images(), 7)
