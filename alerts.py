#Script to send SMS to phone for uidentified users

import logging
import os

import requests

logger = logging.getLogger(__name__)

TIMEOUT = 10


def alert(x):

    #API key is read from the environment, never stored in the repo
    api_key=os.environ.get("FAST2SMS_API_KEY")

    if not api_key:
        logger.warning("FAST2SMS_API_KEY is not set - skipping SMS alert.")
        return

    if not x:
        logger.warning("No alert number configured - skipping SMS alert.")
        return

    url = "https://www.fast2sms.com/dev/bulk"

    payload = "sender_id=FSTSMS&message=Unidentified tried to access system&language=english&route=p&numbers="+x

    headers = {
    'authorization': api_key,
    'Content-Type': "application/x-www-form-urlencoded",'Cache-Control': "no-cache",}

    response = requests.request("POST", url, data=payload, headers=headers, timeout=TIMEOUT)
    response.raise_for_status()
    logger.info("Alert SMS dispatched.")
