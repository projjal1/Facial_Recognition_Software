"""Turn a stream of OpenCV frames into an MJPEG HTTP response body.

multipart/x-mixed-replace is the oldest trick in the book for pushing video to a
browser, and the reason it suits this project is that an <img> tag consumes it
with no JavaScript at all. Each frame is a complete JPEG, so there is no codec
state to keep and a client can join or leave at any point.

The important property is what happens when the viewer navigates away: the
server's write fails, Django stops iterating, the generator is closed, and the
`finally` in the frame source releases the camera. That is what replaces the
old "press q in the window on the server" - which could not work once the
process had no window.
"""

import logging
import textwrap

import cv2
import numpy as np
from django.conf import settings
from django.http import HttpResponse

logger = logging.getLogger(__name__)

BOUNDARY = 'frame'
CONTENT_TYPE = 'multipart/x-mixed-replace; boundary=%s' % BOUNDARY


def mjpeg(frames):
    """Yield multipart chunks, one per frame, until `frames` is exhausted."""
    sent = 0
    try:
        for frame in frames:
            ok, buffer = cv2.imencode(
                '.jpg', frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), settings.STREAM_JPEG_QUALITY])
            if not ok:
                logger.warning("Dropped a frame that would not encode as JPEG.")
                continue

            payload = buffer.tobytes()
            sent += 1
            yield (
                b'--' + BOUNDARY.encode() + b'\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(payload)).encode() + b'\r\n'
                b'\r\n' + payload + b'\r\n'
            )
    except GeneratorExit:
        # The viewer went away. Not an error; let the frame source clean up.
        logger.info("Stream closed by the client after %d frames.", sent)
        raise
    finally:
        logger.info("Stream finished after %d frames.", sent)


def error_image(message, width=640, height=360):
    """A single JPEG carrying `message`, as bytes.

    Stream endpoints are fetched by an <img> tag, so answering with an HTML
    error page leaves the viewer looking at a broken image with no idea what
    went wrong. Drawing the reason into a frame puts it where they are already
    looking.
    """
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = textwrap.wrap(message, width=46) or ['Unavailable']
    y = max(40, height // 2 - (len(lines) * 28) // 2)

    for line in lines:
        cv2.putText(canvas, line, (24, y), font, 0.62, (220, 220, 220), 1,
                    cv2.LINE_AA)
        y += 30

    ok, buffer = cv2.imencode('.jpg', canvas)
    return buffer.tobytes() if ok else b''


def error_response(message):
    """An image/jpeg response for a stream that could not start."""
    logger.info("Stream refused: %s", message)
    return HttpResponse(error_image(message), content_type='image/jpeg')


def primed(frames):
    """Pull the first frame so setup failures surface before streaming starts.

    Once a StreamingHttpResponse has begun there is no way to send an error
    status - the browser is already being handed image data. Drawing the first
    frame here means a missing camera or an unreadable URL still raises inside
    the view, where it can be rendered as a normal page.
    """
    first = next(frames)

    def rest():
        yield first
        for frame in frames:
            yield frame

    return rest()
