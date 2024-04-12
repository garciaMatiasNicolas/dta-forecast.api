from django.urls import path
from .views import SearchProject

search = SearchProject.search_by_name

urlpatterns = [
    path('search-project', search, name='search_project'),
]