from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.views.static import serve

urlpatterns = [
    path('admin/', admin.site.urls),
    path('users/', include('users.urls')),
    path('users/authentication/', include('users.authentication_urls')),
    path('files/', include('files.router')),
    path('projects/', include('projects.router')),
    path('scenarios/', include('forecast.router')),
    path('forecast/', include('forecast.urls')),
    path('', include('projects.urls')),
    path('', include('files.urls')),
    path('media/<path:path>', serve, {'document_root': settings.MEDIA_ROOT}),
]
