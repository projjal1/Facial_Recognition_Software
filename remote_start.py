#This module captures enrolment images from a remote camera URL

import logging

import camera
import enrolment
import face_store

logger = logging.getLogger(__name__)


def record_vid(username, existing, url):
    """Annotated frames enrolling from a remote camera URL."""
    folder = face_store.folder_for(username, create=True)
    # Mirrored to match the local path, so a person sees themselves the same way
    # in both enrolment flows.
    return enrolment.capture(
        camera.remote_frames(url, flip=True), folder, existing)
