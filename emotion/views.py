from django.shortcuts import render,redirect
from emotion.resources import cam

def detect(request):
    cam.capture()
    return redirect("home")
