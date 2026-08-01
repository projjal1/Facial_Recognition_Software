#This module captures enrolment images from a remote camera URL

import logging
import os

from django.conf import settings

import camera
import enrolment

logger = logging.getLogger(__name__)


def record_vid(username, existing, url):
    folder = os.path.join(settings.BASE_DIR, username)
    # Mirrored to match the local path, so a person sees themselves the same way
    # in both enrolment flows.
    return enrolment.capture(
        camera.remote_frames(url, flip=True), folder, existing)
