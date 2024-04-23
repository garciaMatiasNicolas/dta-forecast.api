from django.urls import path
from .views import UserViews

create = UserViews.create_user
detail = UserViews.detail_user
get_all = UserViews.get_all_users

urlpatterns = [
    path('create', create, name='create_user'),
    path('get', get_all, name='get_all_users'),
    path('detail/<int:pk>', detail, name='delete_put_get_user'),
]