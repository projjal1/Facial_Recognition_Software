"""Enrolment capture: what gets written, and what gets refused.

The duplicate check is the part worth pinning down. Enrolling one person under
two names splits their images across two classes and leaves the recogniser
unable to separate them, which surfaces much later as poor accuracy with no
obvious cause - so nothing may reach disk until the identity is settled.
"""

import os
import shutil
import tempfile
from unittest import mock

import numpy as np
from django.test import SimpleTestCase, override_settings

import enrolment


class FakeModel:
    """Answers every prediction the same way, which is all the check needs."""

    def __init__(self, label, distance):
        self.label = label
        self.distance = distance

    def predict(self, crop):
        return self.label, self.distance


def sharp():
    return np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)


def blurred():
    return np.zeros((64, 64), dtype=np.uint8)


def frames(count=30):
    for _ in range(count):
        yield np.zeros((80, 80, 3), dtype=np.uint8)


def detecting(crop):
    return lambda frame: [((0, 0, 64, 64), crop)]


@override_settings(FACE_ENROLMENT_FRAMES=4, FACE_BLUR_THRESHOLD=40,
                   FACE_DUPLICATE_THRESHOLD=45, FACE_DUPLICATE_CHECK_FRAMES=3)
class CaptureTests(SimpleTestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, True)

    def saved(self):
        return sorted(os.listdir(self.folder))

    def run_capture(self, label=None):
        list(enrolment.capture(frames(), self.folder, 0, label=label))

    def test_saves_up_to_the_target(self):
        with mock.patch('faces.crops', detecting(sharp())), \
             mock.patch('recog.load_model', return_value=None):
            self.run_capture()
        self.assertEqual(self.saved(), ['1.jpg', '2.jpg', '3.jpg', '4.jpg'])

    def test_numbers_after_the_existing_images(self):
        with mock.patch('faces.crops', detecting(sharp())), \
             mock.patch('recog.load_model', return_value=None):
            list(enrolment.capture(frames(), self.folder, 10))
        self.assertEqual(self.saved()[0], '11.jpg')

    def test_blurred_crops_are_never_saved(self):
        with mock.patch('faces.crops', detecting(blurred())), \
             mock.patch('recog.load_model', return_value=None):
            self.run_capture()
        self.assertEqual(self.saved(), [])

    def test_an_untrained_system_skips_the_duplicate_check(self):
        # Nothing to compare against, so enrolment must not be blocked.
        with mock.patch('faces.crops', detecting(sharp())), \
             mock.patch('recog.load_model', return_value=None):
            self.run_capture(label=1)
        self.assertEqual(len(self.saved()), 4)

    def test_a_face_already_enrolled_as_someone_else_writes_nothing(self):
        # A close match to label 3 while enrolling as label 1.
        with mock.patch('faces.crops', detecting(sharp())), \
             mock.patch('recog.load_model', return_value=FakeModel(3, 10)):
            self.run_capture(label=1)
        self.assertEqual(self.saved(), [])

    def test_a_distant_match_does_not_block_enrolment(self):
        # Same label, but far beyond the duplicate distance.
        with mock.patch('faces.crops', detecting(sharp())), \
             mock.patch('recog.load_model', return_value=FakeModel(3, 200)):
            self.run_capture(label=1)
        self.assertEqual(len(self.saved()), 4)

    def test_matching_your_own_label_is_not_a_duplicate(self):
        # Enrolling more images of yourself is the normal top-up case.
        with mock.patch('faces.crops', detecting(sharp())), \
             mock.patch('recog.load_model', return_value=FakeModel(1, 5)):
            self.run_capture(label=1)
        self.assertEqual(len(self.saved()), 4)

    def test_held_crops_are_flushed_once_the_identity_clears(self):
        # The crops examined during the check still count toward the target,
        # so a clean run is not short by however many were sampled.
        with mock.patch('faces.crops', detecting(sharp())), \
             mock.patch('recog.load_model', return_value=FakeModel(3, 200)):
            self.run_capture(label=1)
        self.assertEqual(len(self.saved()), 4)

    def test_the_stream_still_produces_frames_while_refusing(self):
        with mock.patch('faces.crops', detecting(sharp())), \
             mock.patch('recog.load_model', return_value=FakeModel(3, 10)):
            produced = list(enrolment.capture(frames(), self.folder, 0, label=1))
        self.assertGreater(len(produced), 0)
