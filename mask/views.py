from django.shortcuts import render,redirect
from mask.resources import webcam

# Create your views here.
def start_mask(request):
    webcam.capture()
    return redirect("home")