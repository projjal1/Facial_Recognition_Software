#This module is used to start the identification process using remote stream

import logging

from django.conf import settings

import camera
import recognition

logger = logging.getLogger(__name__)


def remote(url, names):
    logger.info("Starting remote recognition against %s.", url)
    recognition.run(
        camera.remote_frames(url), names,
        settings.FACE_CONFIDENCE_THRESHOLD_REMOTE)
