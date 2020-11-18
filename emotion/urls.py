from django.urls import path
from . import views

urlpatterns = [    
    path('apps/',views.detector,name='emotion-detect'),
    path('video', views.video_feed, name='vid'),
]