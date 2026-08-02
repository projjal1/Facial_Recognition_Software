"""Project-level views that do not belong to any single app."""

import logging

from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme

import camera

logger = logging.getLogger(__name__)


@login_required
def stop_camera(request):
    """Release the webcam and send the viewer on.

    Every capture page ends with a button pointing here. Without it, leaving a
    page does not stop the work behind it - the stream only ends once a write
    to its socket fails, and until then the detector is still running a model
    over every frame.

    POST only: this changes server state, and a link prefetch should not be
    able to stop someone else's capture.
    """
    if request.method != 'POST':
        return redirect('home')

    if camera.release_current():
        logger.info("Capture stopped by %s.", request.user.username)

    target = request.POST.get('next') or ''
    # An unchecked redirect target is an open redirect; only same-host paths.
    if not url_has_allowed_host_and_scheme(
            target,
            allowed_hosts={request.get_host()},
            require_https=request.is_secure()):
        target = reverse('home')

    return redirect(target)
