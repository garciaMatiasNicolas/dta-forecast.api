from django.db import models
from users.models import User
from projects.models import ProjectsModel


class FileTypes(models.Model):
    model_type = models.CharField(max_length=200)


class FileRefModel(models.Model):
    user_owner = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=200)
    model_type = models.ForeignKey(FileTypes, on_delete=models.CASCADE)
    file = models.FileField(upload_to='excel_files/data')
    project = models.ForeignKey(ProjectsModel, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)




