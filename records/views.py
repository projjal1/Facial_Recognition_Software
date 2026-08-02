import logging
import os

import cv2
import numpy as np
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse
from django.shortcuts import render
from PIL import Image

import face_store
import faces
import remote_start
import start
import streaming
from security import validate_camera_url

logger = logging.getLogger(__name__)

# Where get_face_remote leaves the camera URL for the stream request that
# follows. Kept in the session rather than the query string so the address is
# not written into browser history or server access logs.
REMOTE_URL_KEY = 'enrol_camera_url'


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
    """The page holding the enrolment video; frames come from `stream_local`."""
    try:
        _user_folder(request.user.username)
    except ValueError as exc:
        return render(request, 'page1.html', {'error': str(exc)})

    return render(request, 'live_enrol.html', {'remote': False})


@login_required
def get_face_remote(request):
    try:
        _user_folder(request.user.username)
        url = validate_camera_url(request.POST.get('link', ''))
    except ValueError as exc:
        return render(request, 'page1.html', {'error': str(exc)})

    request.session[REMOTE_URL_KEY] = url
    return render(request, 'live_enrol.html', {'remote': True})


def _stream(request, build_source):
    """Shared plumbing for both enrolment streams."""
    username = request.user.username
    try:
        folder = _user_folder(username)
        source = build_source(username, len(os.listdir(folder)))
        # Draw the first frame here so a missing camera is reported before the
        # response starts, while there is still a way to say so.
        source = streaming.primed(source)
    except ValueError as exc:
        return streaming.error_response(str(exc))
    except StopIteration:
        return streaming.error_response('The camera produced no frames.')
    except Exception:
        logger.exception("Enrolment stream failed to start for %s.", username)
        return streaming.error_response(
            'Could not capture from that camera. Check that it is connected '
            'and not already in use.')

    return StreamingHttpResponse(
        streaming.mjpeg(source), content_type=streaming.CONTENT_TYPE)


@login_required
def stream_local(request):
    return _stream(request, start.record_vid)


@login_required
def stream_remote(request):
    url = request.session.get(REMOTE_URL_KEY)
    if not url:
        return streaming.error_response(
            'No camera URL for this session. Submit the address again.')

    return _stream(
        request,
        lambda username, existing: remote_start.record_vid(username, existing, url))


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

    try:
        # Pillow decodes more formats than cv2.imdecode, which matters for
        # whatever a phone camera produces; convert to BGR for the rest.
        image = Image.open(upload).convert('RGB')
    except OSError:
        logger.exception("Upload from %s could not be decoded.", username)
        return render(request, 'page2.html', {
            'msg': 'That file could not be read as an image.',
        })

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Store the same normalised crop the webcam paths store, rather than the
    # whole photo. Keeping every stored image in one form is what lets training
    # skip detection entirely and guarantees enrolment matches recognition.
    detected = list(faces.crops(frame))
    if not detected:
        return render(request, 'page2.html', {
            'msg': 'No face was found in that image. Take a closer photo, '
                   'well lit, with the face filling most of the frame.',
        })

    _box, crop = detected[0]
    if not faces.is_sharp(crop):
        return render(request, 'page2.html', {
            'msg': 'That photo is too blurred to enrol. Try a sharper one.',
        })

    destination = os.path.join(folder, "img%d.jpg" % (len(os.listdir(folder)) + 1))
    cv2.imwrite(destination, crop)

    return render(request, 'page2.html', {'msg': 'Got your image uploaded'})
