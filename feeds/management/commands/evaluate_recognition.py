"""Measure recognition accuracy and print a threshold table.

The confidence thresholds in settings have always been guesses, and nothing in
the project ever measured what they cost. This holds out part of each person's
images, trains on the rest, and reports what happens at a range of thresholds -
so the value can be chosen against a false-accept rate rather than picked.

It never touches trainer.yml: the model it builds lives in memory for the length
of the run.

    python manage.py evaluate_recognition
    python manage.py evaluate_recognition --holdout 0.4 --max-far 0.005
"""

import os
import random

import cv2
import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

import face_store


class Command(BaseCommand):
    help = "Evaluate face recognition accuracy and suggest a confidence threshold."

    def add_arguments(self, parser):
        parser.add_argument('--holdout', type=float, default=0.3,
                            help="Fraction of each person's images to test on.")
        parser.add_argument('--seed', type=int, default=0,
                            help="Seed for the split, so runs are comparable.")
        parser.add_argument('--max-far', type=float, default=0.01,
                            help="Highest acceptable false-accept rate.")

    def handle(self, *args, **options):
        train, test = self._split(options['holdout'], options['seed'])

        people = len({label for label, _ in train})
        if people < 2:
            raise CommandError(
                "Need at least two enrolled people to measure anything; found "
                "%d. With one person the recogniser matches everybody to them."
                % people)
        if not test:
            raise CommandError(
                "The holdout came out empty. Enrol more images per person, or "
                "raise --holdout.")

        self.stdout.write("%d people, %d training images, %d held out"
                          % (people, len(train), len(test)))

        results = self._score(train, test)
        self._report(results, options['max_far'])

    def _split(self, holdout, seed):
        """Per-person split, so every person appears in both halves."""
        rng = random.Random(seed)
        train, test = [], []
        size = settings.FACE_CROP_SIZE

        for label, folder in face_store.enrolled_folders():
            crops = []
            for name in sorted(os.listdir(folder)):
                crop = cv2.imread(os.path.join(folder, name), cv2.IMREAD_GRAYSCALE)
                if crop is None:
                    continue
                if crop.shape != (size, size):
                    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
                crops.append(crop)

            if len(crops) < 2:
                self.stderr.write("s%d has %d image(s); skipping."
                                  % (label, len(crops)))
                continue

            rng.shuffle(crops)
            cut = max(1, int(len(crops) * holdout))
            test.extend((label, c) for c in crops[:cut])
            train.extend((label, c) for c in crops[cut:])

        return train, test

    def _score(self, train, test):
        """Nearest-label distance for every held-out image."""
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train([c for _, c in train],
                         np.array([l for l, _ in train]))

        results = []
        for true_label, crop in test:
            predicted, distance = recognizer.predict(crop)
            results.append((true_label, predicted, distance))
        return results

    def _report(self, results, max_far):
        total = len(results)
        distances = sorted(d for _, _, d in results)

        # Sweep across the range the data actually occupies rather than an
        # arbitrary span; a threshold outside it tells you nothing.
        low, high = distances[0], distances[-1]
        step = max(1.0, (high - low) / 20.0)

        self.stdout.write("")
        self.stdout.write("%-12s %-10s %-10s %-10s" % (
            "threshold", "correct", "wrong", "rejected"))
        self.stdout.write("-" * 44)

        best = None
        threshold = low
        while threshold <= high + step:
            correct = wrong = rejected = 0
            for true_label, predicted, distance in results:
                if distance >= threshold:
                    rejected += 1
                elif predicted == true_label:
                    correct += 1
                else:
                    wrong += 1

            tar, far = correct / total, wrong / total
            self.stdout.write("%-12.0f %-9.1f%% %-9.1f%% %-9.1f%%" % (
                threshold, tar * 100, far * 100, rejected / total * 100))

            if far <= max_far and (best is None or tar > best[1]):
                best = (threshold, tar, far)
            threshold += step

        self.stdout.write("")
        if best is None:
            self.stdout.write(self.style.WARNING(
                "No threshold keeps the false-accept rate at or below %.1f%%. "
                "The model cannot separate these people - more varied images "
                "per person is the usual fix." % (max_far * 100)))
            return

        threshold, tar, far = best
        self.stdout.write(self.style.SUCCESS(
            "Suggested threshold %d: recognises %.1f%% correctly with a "
            "%.1f%% false-accept rate." % (threshold, tar * 100, far * 100)))
        self.stdout.write(
            "Set FACE_CONFIDENCE_THRESHOLD_LOCAL and _REMOTE to this, in "
            "settings or the environment.")
