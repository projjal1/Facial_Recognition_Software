import logging
import os

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from PIL import Image

import face_store
import remote_start
import start
from security import validate_camera_url

logger = logging.getLogger(__name__)


def _user_folder(username):
    return face_store.folder_for(username, create=True)


@login_required
def face(request):
    try:
        count = len(os.listdir(_user_folder(request.user.username)))
    except ValueError as exc:
        return render(request, 'page1.html', {'error': str(exc)})

    return render(request, 'page1.html', {'msg': str(count) if count else ''})


@login_required
def get_face(request):
    username = request.user.username
    try:
        folder = _user_folder(username)
    except ValueError as exc:
        return render(request, 'page1.html', {'error': str(exc)})

    try:
        start.record_vid(username, len(os.listdir(folder)))
    except Exception:
        # Was unguarded, so a camera that is missing or already in use took the
        # whole request down with a traceback.
        logger.exception("Local capture failed for %s.", username)
        return render(request, 'page2.html', {
            'msg': 'Could not capture from the webcam. Check that it is '
                   'connected and not already in use.',
        })

    return render(request, 'page2.html', {'msg': 'Fetched images'})


@login_required
def get_face_remote(request):
    username = request.user.username

    # UnsafeCameraURL subclasses ValueError, so both failures land here.
    try:
        folder = _user_folder(username)
        url = validate_camera_url(request.POST.get('link', ''))
    except ValueError as exc:
        return render(request, 'page1.html', {'error': str(exc)})

    try:
        remote_start.record_vid(username, len(os.listdir(folder)), url)
    except Exception:
        # Previously `except: pass`, which reported success after a failure.
        logger.exception("Remote capture failed for %s.", username)
        return render(request, 'page2.html', {
            'msg': 'Could not read from that camera URL. Check the address '
                   'and that the device is reachable.',
        })

    return render(request, 'page2.html', {'msg': 'Fetched images'})


@login_required
def fetch(request):
    username = request.user.username
    try:
        folder = _user_folder(username)
    except ValueError as exc:
        return render(request, 'page1.html', {'error': str(exc)})

    upload = request.FILES.get('id_image')
    if upload is None:
        return render(request, 'page2.html', {'msg': 'No image was selected.'})

    destination = os.path.join(folder, "img%d.jpg" % (len(os.listdir(folder)) + 1))

    try:
        # Convert first: a PNG with transparency cannot be written as JPEG.
        Image.open(upload).convert('RGB').save(destination)
    except OSError:
        logger.exception("Upload from %s could not be decoded.", username)
        return render(request, 'page2.html', {
            'msg': 'That file could not be read as an image.',
        })

    return render(request, 'page2.html', {'msg': 'Got your image uploaded'})
