"""Reads and writes for the flat-file state under admin_files/.

This project keeps its runtime state - the entry log, the alert number, the
camera URL, the trained-image count - in plain text rather than the database.
Centralising the access keeps path handling in one place, and lets a missing
file read as empty instead of raising: these are runtime artefacts, so a fresh
checkout legitimately has none of them populated and that should not be a 500.
"""

import logging
import os

from django.conf import settings

logger = logging.getLogger(__name__)

LOGS = 'logs.txt'
MOBILE_NO = 'mobile_no.txt'
LINK = 'link.txt'
TRAINED = 'trained.txt'


def path(name):
    return os.path.join(settings.BASE_DIR, 'admin_files', name)


def read(name):
    try:
        with open(path(name), 'r') as handle:
            return handle.read()
    except FileNotFoundError:
        logger.info("admin_files/%s does not exist yet; treating as empty.", name)
        return ''


def _write(name, content, mode):
    target = path(name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, mode) as handle:
        handle.write(content)


def write(name, content):
    _write(name, content, 'w')


def append(name, content):
    _write(name, content, 'a')


def read_int(name, default=0):
    """Last integer in the file, or the default when absent or unparseable."""
    value = read(name).strip()
    if not value:
        return default
    try:
        return int(value.splitlines()[-1])
    except ValueError:
        logger.warning("admin_files/%s did not contain a number.", name)
        return default
