from django.contrib import admin 
from django.urls import path,include
from emotion import views

urlpatterns = [
    path('apps/', views.detect,name='emotion'),
]
