import requests

def alert(x):

    url = "https://www.fast2sms.com/dev/bulk"

    payload = "sender_id=FSTSMS&message=Unidentified tried to access system&language=english&route=p&numbers="+x

    headers = {
    'authorization': "Use your own auth-id here",
    'Content-Type': "application/x-www-form-urlencoded",'Cache-Control': "no-cache",}

    response = requests.request("POST", url, data=payload, headers=headers)
