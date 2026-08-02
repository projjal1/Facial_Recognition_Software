import logging
import os
import re

from django.conf import settings
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.http import StreamingHttpResponse
from django.shortcuts import render

import admin_state
import face_store
import identify
import recog
import recognition
import streaming
import webcam
from security import validate_camera_url

logger = logging.getLogger(__name__)

superuser_required = user_passes_test(lambda user: user.is_superuser)


def _label_names():
    """Map each LBPH label to the username that owns it.

    The trainer derives a person's label from the digits in their folder name,
    so this builds the lookup the same way. The previous version appended
    usernames to a list in database order and indexed into it, which meant a
    deleted account or a gap in the numbering silently shifted every name - and
    a label past the end of the list raised IndexError mid-capture.
    """
    names = {}
    for user in User.objects.all():
        if re.match(settings.FACE_USERNAME_PATTERN, user.username):
            names[int(user.username[1:])] = user.username
    return names


def _training_status():
    pending = abs(face_store.total_images() - admin_state.read_int(admin_state.TRAINED))
    if pending == 0:
        return "All data have been previously trained. You can skip with training process."
    return "You have %d pending data. You should train the model" % pending


def _not_ready():
    """Why detection cannot start yet, or None when it can.

    Checked before rendering the page rather than only inside the stream: once
    a streaming response has begun there is no way to show an error, so the
    viewer would just get a broken image with no explanation.
    """
    if not _label_names():
        return ('No enrolled accounts yet. Sign up as s1 and register a face '
                'before starting detection.')
    if not os.path.exists(recognition.MODEL_PATH):
        return 'No trained model yet. Train the model before starting detection.'
    return None


@login_required
@superuser_required
def detection(request):
    return render(request, "start_captures.html")


@login_required
@superuser_required
def init_url(request):
    try:
        url = validate_camera_url(request.POST.get('url', ''))
    except ValueError as exc:
        return render(request, "start_captures.html", {'error': str(exc)})

    admin_state.write(admin_state.LINK, url)
    return render(request, "model_detection.html", {'msg': _training_status()})


@login_required
@superuser_required
def init_server(request):
    admin_state.write(admin_state.LINK, '')
    return render(request, "model_detection.html", {'msg': _training_status()})


@login_required
@superuser_required
def start(request):
    """The page that holds the video. The frames come from `stream` below."""
    problem = _not_ready()
    if problem:
        return render(request, "start_captures.html", {'error': problem})

    source = admin_state.read(admin_state.LINK).strip()
    return render(request, "live_detect.html", {
        'source': source or 'the server webcam',
    })


@login_required
@superuser_required
def stream(request):
    """The MJPEG body consumed by the <img> tag on the detection page."""
    problem = _not_ready()
    if problem:
        return streaming.error_response(problem)

    names = _label_names()
    url = admin_state.read(admin_state.LINK).strip()

    try:
        if url:
            # Re-checked rather than trusting what is on disk, since the file is
            # plain text that anything could have written.
            source = webcam.remote(validate_camera_url(url), names)
        else:
            source = identify.captures(names)

        # Draw the first frame here so a missing camera is reported before the
        # response starts, while there is still a way to say so.
        source = streaming.primed(source)
    except ValueError as exc:
        return streaming.error_response(str(exc))
    except StopIteration:
        return streaming.error_response('The camera produced no frames.')
    except Exception:
        logger.exception("Detection stream failed to start.")
        return streaming.error_response(
            'Could not start detection. Check the camera and the log.')

    return StreamingHttpResponse(
        streaming.mjpeg(source), content_type=streaming.CONTENT_TYPE)


@login_required
@superuser_required
def end(request):
    return render(request, "start_captures.html")


@login_required
@superuser_required
def train(request):
    # Incremental by default, since people enrol one at a time. A rebuild is
    # needed after the stored crop format changes, or after images are deleted -
    # LBPH cannot forget, so anything subtractive means starting over.
    full = request.POST.get('full') == 'on'

    try:
        recog.begin(full=full)
    except ValueError as exc:
        return render(request, "model_detection.html", {'msg': str(exc)})
    except Exception:
        logger.exception("Training failed.")
        return render(request, "model_detection.html", {
            'msg': "Training failed. Check the server log for details.",
        })

    return render(request, "model_detection.html", {'msg': "Training over."})
