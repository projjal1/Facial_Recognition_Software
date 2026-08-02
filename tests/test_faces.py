"""Detection and crop normalisation.

Detection itself needs a real face, which no synthetic image supplies, so what
is asserted here is the part that governs whether enrolment and recognition
agree: that a crop always comes out at the configured size, equalised, and that
blurred crops are recognisably blurred.
"""

import cv2
import numpy as np
from django.test import SimpleTestCase, override_settings

import faces


def frame(width=400, height=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


class NormaliseTests(SimpleTestCase):

    @override_settings(FACE_CROP_SIZE=128)
    def test_always_returns_the_configured_size(self):
        # Fixed size is what keeps LBPH's per-cell histograms comparable; crops
        # of varying size distort the grid it divides an image into.
        for box in [(0, 0, 40, 40), (10, 10, 200, 150), (5, 5, 90, 300)]:
            with self.subTest(box=box):
                crop = faces.normalise(frame(), box)
                self.assertEqual(crop.shape, (128, 128))

    @override_settings(FACE_CROP_SIZE=64)
    def test_the_size_follows_the_setting(self):
        self.assertEqual(faces.normalise(frame(), (0, 0, 40, 40)).shape, (64, 64))

    def test_returns_a_single_channel_image(self):
        self.assertEqual(faces.normalise(frame(), (0, 0, 40, 40)).ndim, 2)

    def test_equalisation_spreads_a_flat_crop(self):
        # A low-contrast region should come back with its range opened up,
        # which is the point of running CLAHE over side-lit faces.
        flat = np.full((100, 100, 3), 120, dtype=np.uint8)
        flat[40:60, 40:60] = 130
        crop = faces.normalise(flat, (0, 0, 100, 100))
        self.assertGreater(crop.max() - crop.min(), 10)


class SharpnessTests(SimpleTestCase):

    @override_settings(FACE_BLUR_THRESHOLD=40)
    def test_noise_is_sharp_and_a_flat_crop_is_not(self):
        rng = np.random.default_rng(0)
        self.assertTrue(faces.is_sharp(
            rng.integers(0, 255, (128, 128), dtype=np.uint8)))
        self.assertFalse(faces.is_sharp(np.zeros((128, 128), dtype=np.uint8)))

    @override_settings(FACE_BLUR_THRESHOLD=40)
    def test_blurring_a_sharp_crop_makes_it_fail(self):
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 255, (128, 128), dtype=np.uint8)
        self.assertTrue(faces.is_sharp(sharp))
        self.assertFalse(faces.is_sharp(cv2.GaussianBlur(sharp, (21, 21), 0)))


class DetectTests(SimpleTestCase):

    def test_noise_contains_no_faces(self):
        self.assertEqual(faces.detect(frame()), [])

    @override_settings(FACE_MIN_SIZE=10000)
    def test_the_minimum_size_filters_everything_out(self):
        # A face below the minimum carries no usable identity signal; the old
        # Haar call accepted them down to ten pixels.
        self.assertEqual(faces.detect(frame()), [])

    def test_crops_yields_nothing_when_nothing_is_detected(self):
        self.assertEqual(list(faces.crops(frame())), [])
