from django.db import models
from users.models import User


class ProjectsModel(models.Model):
    user_owner = models.ForeignKey(User, on_delete=models.CASCADE)
    project_name = models.CharField(max_length=100, unique=True)
    status = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)