from django.db import models
from django.contrib.auth.models import BaseUserManager, AbstractBaseUser


class UserManager(BaseUserManager):
    def create_user(self, email: str, first_name: str, last_name: str, password: str = None) -> object:
        if not email:
            raise ValueError('user must have an email')

        user = self.model(
            email=email,
            first_name=first_name,
            last_name=last_name
        )

        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, email: str, first_name: str, last_name: str,
                         password: str = None) -> object:
        admin = self.create_user(
            email=email,
            first_name=first_name,
            last_name=last_name,
            password=password
        )

        admin.is_admin = True
        admin.save()
        return admin


class User(AbstractBaseUser):
    email = models.EmailField(max_length=200, unique=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    email_is_confirmed = models.BooleanField(default=False)
    is_admin = models.BooleanField(default=False)
    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ('first_name', 'last_name')

    def has_perm(self, perm, obj=None) -> bool:
        return True

    def has_module_perms(self, app_label) -> bool:
        return True

    @property
    def is_staff(self) -> object:
        return self.is_admin

    def __str__(self) -> str:
        return f'{self.first_name} {self.last_name}'