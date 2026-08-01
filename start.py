#This module captures enrolment images from the webcam attached to the server

import logging

import camera
import enrolment
import face_store

logger = logging.getLogger(__name__)


def record_vid(username, existing):
    """Annotated frames enrolling from the webcam attached to the server."""
    folder = face_store.folder_for(username, create=True)
    return enrolment.capture(camera.local_frames(), folder, existing)
