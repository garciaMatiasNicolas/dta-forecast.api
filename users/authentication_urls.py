from django.urls import path
from .authentication import UserAuthenticationViews

login = UserAuthenticationViews.LogInView.as_view()
logout = UserAuthenticationViews.LogOutView.as_view()
confirm_email = UserAuthenticationViews.get_confirmation_email

urlpatterns = [
    path('login', login, name='login_view'),
    path('logout', logout, name='logout_view'),
    path('confirm-email/<int:pk>', confirm_email, name='confirm_mail')
]