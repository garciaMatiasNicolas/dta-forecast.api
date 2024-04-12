from rest_framework.routers import DefaultRouter
from .views import ExcelFileUploadView

router_file = DefaultRouter()

router_file.register('', ExcelFileUploadView, basename='file_route')

urlpatterns = router_file.urls