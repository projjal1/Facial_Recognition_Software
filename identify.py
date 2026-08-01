#This module is used to start identification using local webcam

import logging

from django.conf import settings

import camera
import recognition

logger = logging.getLogger(__name__)


def captures(names):
    logger.info("Starting local recognition for %d enrolled people.", len(names))
    recognition.run(
        camera.local_frames(), names, settings.FACE_CONFIDENCE_THRESHOLD_LOCAL)
