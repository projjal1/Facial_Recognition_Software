from django.contrib import admin
from django.urls import path,include
from accounts import views
from chatapp import views as project_views

urlpatterns = [
    path('admin/', admin.site.urls,name='admin'),
    path('',views.base,name='home'),
    path('camera/stop/',project_views.stop_camera,name='camera-stop'),
    path('accounts/',include('accounts.urls')),
    path('reg/',include('records.urls')),
    path('feed_detect/',include('feeds.urls')),
    path('emotion/',include('emotion.urls')),
    path('mask/',include('mask.urls'))
]
