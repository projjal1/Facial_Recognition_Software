import logging
import re

from django.conf import settings
from django.contrib import auth
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.shortcuts import redirect, render

import admin_state
import face_store

logger = logging.getLogger(__name__)

# The navbar hides the admin links, but hiding a link is not access control -
# the URLs answered to anyone before this was added.
superuser_required = user_passes_test(lambda user: user.is_superuser)


def base(request):
    return render(request, "home.html")


def login(request):
    if request.method == "POST":
        user = auth.authenticate(
            username=request.POST['username'],
            password=request.POST['pass1'],
        )
        if user is not None:
            auth.login(request, user)
            return redirect("home")
        return render(request, "log.html",
                      {'error': 'User does not exist or password is wrong.'})
    return render(request, "log.html")


def logout(request):
    # Only POST actually logs out, so a link prefetch cannot end a session. A
    # GET used to fall off the end of the function and return None, which Django
    # raises on; send it home instead.
    if request.method == "POST":
        auth.logout(request)
    return redirect("home")


def signup(request):
    if request.method != "POST":
        return render(request, 'sign.html')

    if request.POST['pass1'] != request.POST['pass2']:
        return render(request, 'sign.html',
                      {'error': 'Sorry, Passwords do not match.'})

    username = request.POST['username']
    password = request.POST['pass1']
    first_name = request.POST['first']
    last_name = request.POST['last']

    if not (username and password and first_name and last_name):
        return render(request, 'sign.html', {'error': '* Fill all details *'})

    # The trainer derives each person's numeric label from their folder name, so
    # a username outside this shape is never trained and the account silently
    # never gets recognised. Rejecting it here turns that into a visible error.
    # It also means the value below is safe to use as a path component.
    if not re.match(settings.FACE_USERNAME_PATTERN, username):
        return render(request, 'sign.html', {
            'error': 'Username must be the letter s followed by a number, '
                     'for example s1 or s12.',
        })

    if User.objects.filter(username=username).exists():
        return render(request, 'sign.html',
                      {'error': 'Sorry, Username already taken.'})

    user = User.objects.create_user(
        username, password=password, first_name=first_name, last_name=last_name)
    auth.login(request, user)

    # Previously os.system("mkdir " + username), which handed an unvalidated
    # POST field to the shell.
    face_store.folder_for(username, create=True)
    logger.info("Created account and image folder for %s.", username)

    return redirect("home")


@login_required
@superuser_required
def profile(request):
    if request.method == "POST":
        admin_state.write(admin_state.MOBILE_NO, request.POST['mobile'])
        logger.info("Alert number updated by %s.", request.user.username)

    return render(request, 'profile.html',
                  {'no': admin_state.read(admin_state.MOBILE_NO)})


@login_required
@superuser_required
def logs(request):
    if request.method == "POST":
        admin_state.write(admin_state.LOGS, '')
        logger.info("Entry log cleared by %s.", request.user.username)

    return render(request, 'logs.html',
                  {'data': admin_state.read(admin_state.LOGS).splitlines()})


@login_required
def about(request):
    username = request.user.username

    if request.method == 'POST':
        user = request.user
        user.first_name = request.POST['fname']
        user.last_name = request.POST['lname']
        user.save()

    return render(request, 'about.html', {
        'fn': request.user.first_name,
        'ln': request.user.last_name,
        'record': face_store.image_count(username),
    })
