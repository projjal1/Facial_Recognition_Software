import logging

from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from mask.resources import webcam

logger = logging.getLogger(__name__)


@login_required
def start_mask(request):
    # This opens the camera on the server, so it needs an account behind it even
    # though the feature itself is not admin-only.
    try:
        webcam.capture()
    except Exception:
        logger.exception("Mask detection failed.")
        return render(request, "home.html", {
            'error': 'Could not start the camera for mask detection.',
        })

    return redirect("home")
