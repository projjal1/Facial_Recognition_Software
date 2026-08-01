"""Validation for operator-supplied camera URLs.

Both the enrolment flow (records/) and the recognition flow (feeds/) accept a
URL that the *server* then fetches. Unvalidated, that is a server-side request
forgery primitive: the caller chooses any address reachable from the host,
including services a browser could never reach directly.

The checks here are deliberately proportionate rather than maximal. An IP
webcam on the local network is the normal deployment for this project, so
blanket-blocking private ranges would break the feature it is meant to protect.
Instead we reject the things that are never a camera - non-HTTP schemes,
loopback, and link-local (which covers the 169.254.169.254 cloud metadata
endpoint) - and leave a settings hook for deployments that want to pin an
explicit allowlist.
"""

import ipaddress
import socket
from urllib.parse import urlparse

from django.conf import settings


class UnsafeCameraURL(ValueError):
    """Raised when a supplied camera URL must not be fetched."""


def _resolved_addresses(host):
    """Every IP the host resolves to, so a name cannot point somewhere else."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise UnsafeCameraURL("Could not resolve '%s'." % host)

    addresses = []
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    if not addresses:
        raise UnsafeCameraURL("Could not resolve '%s' to an IP address." % host)
    return addresses


def validate_camera_url(url):
    """Return the URL stripped, or raise UnsafeCameraURL explaining the refusal.

    The message is written to be shown to the operator, so it says what to fix.
    """
    if not url or not url.strip():
        raise UnsafeCameraURL("No camera URL was supplied.")

    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise UnsafeCameraURL("Camera URL must start with http:// or https://.")

    host = parsed.hostname
    if not host:
        raise UnsafeCameraURL("Camera URL does not contain a host name.")

    # An explicit allowlist, when configured, is the whole check.
    allowed = getattr(settings, "CAMERA_URL_ALLOWED_HOSTS", None) or []
    if allowed:
        if host not in allowed:
            raise UnsafeCameraURL(
                "'%s' is not in CAMERA_URL_ALLOWED_HOSTS." % host
            )
        return url

    for address in _resolved_addresses(host):
        if address.is_loopback:
            raise UnsafeCameraURL(
                "Refusing to fetch a loopback address - that is this server, "
                "not a camera."
            )
        if address.is_link_local:
            raise UnsafeCameraURL(
                "Refusing to fetch a link-local address."
            )

    return url
