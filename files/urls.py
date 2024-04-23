from django.urls import path
from .views import GetFileTypes
from .export_excel import ExportExcelAPIView

file_types = GetFileTypes.as_view()
export_excel = ExportExcelAPIView.as_view()

urlpatterns = [
    path('file-types', file_types, name='file_types'),
    path('export_excel', export_excel, name='export_excel')
]