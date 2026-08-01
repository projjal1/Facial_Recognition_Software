#This module captures enrolment images from the webcam attached to the server

import logging
import os

from django.conf import settings

import camera
import enrolment

logger = logging.getLogger(__name__)


def record_vid(username, existing):
    folder = os.path.join(settings.BASE_DIR, username)
    return enrolment.capture(camera.local_frames(), folder, existing)
