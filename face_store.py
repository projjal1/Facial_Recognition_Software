"""Where enrolled face images live on disk.

Each person's images sit in face-files/<username>/ rather than in a folder at
the project root. Two things fall out of that. The whole set is covered by a
single ignore rule, so biometric data cannot be committed by adding a new
person. And the trainer scans one directory instead of filtering every entry at
the top level, which is what used to require special cases for static/ and
staticfiles/ - a filter that would equally have skipped a legitimately named
folder.

Every path here is built from a validated username, so a name that reached this
module cannot contain a separator or a parent reference.
"""

import logging
import os
import re

from django.conf import settings

logger = logging.getLogger(__name__)


def root():
    return settings.FACE_IMAGE_ROOT


def folder_for(username, create=False):
    """Absolute path to one person's image folder.

    Raises ValueError for a username the trainer could never read - which is
    also what makes the result safe to use as a path.
    """
    if not re.match(settings.FACE_USERNAME_PATTERN, username):
        raise ValueError(
            "Account '%s' cannot enrol faces: images are stored per person in "
            "folders named s followed by a number." % username)

    path = os.path.join(root(), username)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def label_for(username):
    """The numeric label a username owns, validating the name on the way."""
    folder_for(username)
    return int(username[1:])


def enrolled_folders():
    """Yield (label, path) for every enrolled person, ordered by label.

    Sorted numerically rather than by name so s3 precedes s12. Training does
    not depend on the order, but a predictable one makes the logs readable.
    """
    try:
        entries = os.listdir(root())
    except FileNotFoundError:
        # Nobody has enrolled yet; that is not an error.
        return

    found = []
    for entry in entries:
        if not re.match(settings.FACE_USERNAME_PATTERN, entry):
            continue
        path = os.path.join(root(), entry)
        if os.path.isdir(path):
            found.append((int(entry[1:]), path))

    for label, path in sorted(found):
        yield label, path


def image_count(username):
    """How many images a person has, tolerating a folder that does not exist."""
    try:
        return len(os.listdir(folder_for(username)))
    except (FileNotFoundError, ValueError):
        return 0


def total_images():
    """Images across everyone, used to report how much training is pending."""
    return sum(len(os.listdir(path)) for _, path in enrolled_folders())
