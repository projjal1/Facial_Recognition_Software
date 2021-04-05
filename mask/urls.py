from django.contrib import admin 
from django.urls import path,include
from mask import views

urlpatterns = [
    path('apps/', views.start_mask,name='face_mask'),
]
