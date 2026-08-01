import logging
import os

from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse
from django.shortcuts import render
from PIL import Image

import face_store
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
        # Draw the first frame here so a missing camera is still an error page,
        # rather than a broken image with no explanation.
        source = streaming.primed(source)
    except ValueError as exc:
        return render(request, 'page1.html', {'error': str(exc)})
    except StopIteration:
        return render(request, 'page1.html', {
            'error': 'The camera produced no frames.',
        })
    except Exception:
        logger.exception("Enrolment stream failed to start for %s.", username)
        return render(request, 'page1.html', {
            'error': 'Could not capture from that camera. Check that it is '
                     'connected and not already in use.',
        })

    return StreamingHttpResponse(
        streaming.mjpeg(source), content_type=streaming.CONTENT_TYPE)


@login_required
def stream_local(request):
    return _stream(request, start.record_vid)


@login_required
def stream_remote(request):
    url = request.session.get(REMOTE_URL_KEY)
    if not url:
        return render(request, 'page1.html', {
            'error': 'No camera URL for this session. Submit the address again.',
        })

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
