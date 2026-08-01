#This module is used to start identification using local webcam

import logging

from django.conf import settings

import camera
import recognition

logger = logging.getLogger(__name__)


def captures(names):
    """Annotated frames recognising against the webcam attached to the server."""
    logger.info("Starting local recognition for %d enrolled people.", len(names))
    return recognition.frames(
        camera.local_frames(), names, settings.FACE_CONFIDENCE_THRESHOLD_LOCAL)
