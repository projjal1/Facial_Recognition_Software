"""Training: what goes into the model, and what triggers a rebuild.

LBPH can be appended to but never subtracted from, so the interesting cases are
the ones that decide between adding and starting over. The crops here are
noise rather than faces - training does no detection, so anything of the right
shape exercises the same code.
"""

import os
import shutil
import tempfile

import cv2
import numpy as np
from django.test import SimpleTestCase, override_settings

import admin_state
import recog


class TrainingTestCase(SimpleTestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.store = os.path.join(self.root, 'face-files')
        self.model = os.path.join(self.root, 'trainer.yml')

        self.override = override_settings(BASE_DIR=self.root,
                                          FACE_IMAGE_ROOT=self.store,
                                          FACE_CROP_SIZE=64)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(shutil.rmtree, self.root, True)

        # Otherwise training writes trainer.yml into the repository root.
        original = recog.MODEL_PATH
        recog.MODEL_PATH = self.model
        self.addCleanup(setattr, recog, 'MODEL_PATH', original)

        self.rng = np.random.default_rng(0)

    def enrol(self, username, count, start=1):
        folder = os.path.join(self.store, username)
        os.makedirs(folder, exist_ok=True)
        for i in range(start, start + count):
            crop = self.rng.integers(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(os.path.join(folder, '%d.jpg' % i), crop)

    def manifest(self):
        return recog._read_manifest()


class FullTrainingTests(TrainingTestCase):

    def test_refuses_when_nobody_is_enrolled(self):
        with self.assertRaises(ValueError):
            recog.begin()

    def test_writes_a_model_and_a_manifest(self):
        self.enrol('s1', 3)
        self.enrol('s2', 3)
        recog.begin()

        self.assertTrue(os.path.exists(self.model))
        self.assertEqual({1: 3, 2: 3},
                         {k: len(v) for k, v in self.manifest().items()})

    def test_records_the_total_for_the_pending_count(self):
        self.enrol('s1', 4)
        recog.begin()
        self.assertEqual(admin_state.read_int(admin_state.TRAINED), 4)


class IncrementalTrainingTests(TrainingTestCase):

    def test_a_second_run_with_nothing_new_leaves_the_model_alone(self):
        self.enrol('s1', 2)
        self.enrol('s2', 2)
        recog.begin()
        before = os.path.getmtime(self.model)

        recog.begin()
        self.assertEqual(os.path.getmtime(self.model), before)

    def test_new_images_are_added_to_the_manifest(self):
        self.enrol('s1', 2)
        self.enrol('s2', 2)
        recog.begin()

        self.enrol('s1', 3, start=3)
        recog.begin()
        self.assertEqual(len(self.manifest()[1]), 5)

    def test_a_new_person_is_added_without_a_rebuild(self):
        self.enrol('s1', 2)
        self.enrol('s2', 2)
        recog.begin()

        self.enrol('s3', 2)
        recog.begin()
        self.assertIn(3, self.manifest())
        self.assertEqual(admin_state.read_int(admin_state.TRAINED), 6)

    def test_the_model_stays_loadable_after_an_update(self):
        self.enrol('s1', 2)
        self.enrol('s2', 2)
        recog.begin()
        self.enrol('s3', 2)
        recog.begin()

        recognizer = recog.load_model()
        self.assertIsNotNone(recognizer)
        label, _distance = recognizer.predict(
            self.rng.integers(0, 255, (64, 64), dtype=np.uint8))
        self.assertIn(label, [1, 2, 3])


class RebuildTests(TrainingTestCase):

    def test_removing_images_forces_a_rebuild(self):
        # LBPH cannot forget, so anything subtractive means starting over.
        self.enrol('s1', 3)
        self.enrol('s2', 3)
        recog.begin()

        os.remove(os.path.join(self.store, 's1', '1.jpg'))
        recog.begin()
        self.assertEqual(len(self.manifest()[1]), 2)

    def test_a_missing_manifest_forces_a_rebuild(self):
        self.enrol('s1', 2)
        self.enrol('s2', 2)
        recog.begin()

        admin_state.write(recog.MANIFEST, '')
        self.enrol('s1', 1, start=9)
        recog.begin()
        self.assertEqual(len(self.manifest()[1]), 3)

    def test_an_unreadable_manifest_is_treated_as_missing(self):
        self.enrol('s1', 2)
        admin_state.write(recog.MANIFEST, 'not json')
        recog.begin()
        self.assertEqual(len(self.manifest()[1]), 2)

    def test_full_rebuilds_on_request(self):
        self.enrol('s1', 2)
        self.enrol('s2', 2)
        recog.begin()
        before = os.path.getmtime(self.model)

        recog.begin(full=True)
        self.assertGreaterEqual(os.path.getmtime(self.model), before)
        self.assertEqual(sum(len(v) for v in self.manifest().values()), 4)


class LoadModelTests(TrainingTestCase):

    def test_returns_none_before_anything_is_trained(self):
        self.assertIsNone(recog.load_model())
