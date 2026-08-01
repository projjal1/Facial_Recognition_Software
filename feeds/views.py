import logging
import os
import re

from django.conf import settings
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.shortcuts import render

import admin_state
import identify
import recog
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


def _enrolled_image_count():
    """Total images across every face folder.

    Matching on s<number> rather than a leading 's' is what lets static/ and
    staticfiles/ stop being special cases - they simply do not match.
    """
    total = 0
    for entry in os.listdir(settings.BASE_DIR):
        if not re.match(settings.FACE_USERNAME_PATTERN, entry):
            continue
        folder = os.path.join(settings.BASE_DIR, entry)
        if os.path.isdir(folder):
            total += len(os.listdir(folder))
    return total


def _training_status():
    pending = abs(_enrolled_image_count() - admin_state.read_int(admin_state.TRAINED))
    if pending == 0:
        return "All data have been previously trained. You can skip with training process."
    return "You have %d pending data. You should train the model" % pending


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
    names = _label_names()
    if not names:
        return render(request, "start_captures.html", {
            'error': 'No enrolled accounts yet. Sign up as s1 and register a '
                     'face before starting detection.',
        })

    url = admin_state.read(admin_state.LINK).strip()

    try:
        if url:
            # Re-checked here rather than trusting what is on disk, since the
            # file is plain text that anything could have written.
            webcam.remote(validate_camera_url(url), names)
        else:
            identify.captures(names)
    except ValueError as exc:
        return render(request, "start_captures.html", {'error': str(exc)})
    except Exception:
        # Previously `except:` returning a fixed "check url name" message, which
        # hid camera and model failures behind a misleading explanation.
        logger.exception("Detection run failed.")
        return render(request, "start_captures.html", {
            'error': 'Could not start detection. Check the camera, and that '
                     'the model has been trained.',
        })

    return render(request, "model_detection.html")


@login_required
@superuser_required
def end(request):
    return render(request, "start_captures.html")


@login_required
@superuser_required
def train(request):
    try:
        recog.begin()
    except ValueError as exc:
        return render(request, "model_detection.html", {'msg': str(exc)})
    except Exception:
        logger.exception("Training failed.")
        return render(request, "model_detection.html", {
            'msg': "Training failed. Check the server log for details.",
        })

    return render(request, "model_detection.html", {'msg': "Training over."})
