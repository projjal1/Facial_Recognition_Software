import logging

from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse
from django.shortcuts import render

import camera
import streaming
from emotion.resources import cam

logger = logging.getLogger(__name__)


@login_required
def detect(request):
    """The page that holds the video. Frames come from `stream` below."""
    return render(request, 'live_emotion.html')


@login_required
def stream(request):
    try:
        # Drawing the first frame here means a camera that is missing or busy
        # is still an error page, not a broken image with no explanation.
        source = streaming.primed(cam.frames(camera.local_frames()))
    except ValueError as exc:
        return render(request, 'live_emotion.html', {'error': str(exc)})
    except StopIteration:
        return render(request, 'live_emotion.html', {
            'error': 'The camera produced no frames.',
        })
    except Exception:
        logger.exception("Emotion stream failed to start.")
        return render(request, 'live_emotion.html', {
            'error': 'Could not start the camera for emotion detection.',
        })

    return StreamingHttpResponse(
        streaming.mjpeg(source), content_type=streaming.CONTENT_TYPE)
