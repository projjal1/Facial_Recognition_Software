import logging

from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse
from django.shortcuts import render

import camera
import streaming
from mask.resources import webcam

logger = logging.getLogger(__name__)


@login_required
def start_mask(request):
    """The page that holds the video. Frames come from `stream` below."""
    return render(request, 'live_mask.html')


@login_required
def stream(request):
    try:
        # Drawing the first frame here means a camera that is missing or busy
        # is still an error page, not a broken image with no explanation.
        source = streaming.primed(webcam.frames(camera.local_frames()))
    except ValueError as exc:
        return render(request, 'live_mask.html', {'error': str(exc)})
    except StopIteration:
        return render(request, 'live_mask.html', {
            'error': 'The camera produced no frames.',
        })
    except Exception:
        logger.exception("Mask stream failed to start.")
        return render(request, 'live_mask.html', {
            'error': 'Could not start the camera for mask detection.',
        })

    return StreamingHttpResponse(
        streaming.mjpeg(source), content_type=streaming.CONTENT_TYPE)
