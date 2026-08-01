import logging

from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from emotion.resources import cam

logger = logging.getLogger(__name__)


@login_required
def detect(request):
    # This opens the camera on the server, so it needs an account behind it even
    # though the feature itself is not admin-only.
    try:
        cam.capture()
    except Exception:
        logger.exception("Emotion detection failed.")
        return render(request, "home.html", {
            'error': 'Could not start the camera for emotion detection.',
        })

    return redirect("home")
