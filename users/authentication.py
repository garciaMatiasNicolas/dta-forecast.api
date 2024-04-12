from datetime import datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken
from django.contrib.sessions.models import Session
from .models import User
from django.shortcuts import redirect
from django.core.exceptions import ObjectDoesNotExist


class UserAuthenticationViews:
    @staticmethod
    @api_view(['GET'])
    def get_confirmation_email(request, pk: int):
        if request.method == "GET":
            user = User.objects.get(id=pk)

            if user:
                user.email_is_confirmed = True
                user.save()
                return redirect("http://localhost:3000/confirmation")

    class LogInView(ObtainAuthToken):
        def post(self, request, *args, **kwargs):
            credentials = self.get_serializer(data=request.data,
                                              context={'request': request})

            email = credentials.initial_data['username']
            try:
                User.objects.get(email=email)
            except ObjectDoesNotExist:
                return Response({'error': 'user_not_found'}, status=status.HTTP_404_NOT_FOUND)

            if credentials.is_valid():
                user = credentials.validated_data['user']

                if not user.email_is_confirmed:
                    return Response({'error': 'user_inactive'},
                                    status=status.HTTP_401_UNAUTHORIZED)

                token, created = Token.objects.get_or_create(user=user)
                if created:
                    return Response({
                        'token': token.key,
                        'user_id': user.pk,
                        'email': user.email
                    }, status=status.HTTP_200_OK)

                else:
                    token.delete()
                    new_token = Token.objects.create(user=user)
                    return Response({
                        'token': new_token.key,
                        'user_id': user.pk
                    }, status=status.HTTP_200_OK)

            else:
                return Response({'error': 'credentials_invalids'},
                                status=status.HTTP_400_BAD_REQUEST)

    class LogOutView(APIView):

        def post(self, request):

            try:
                request_token = request.GET.get("token")
                token = Token.objects.filter(key=request_token).first()
                if token:
                    user = token.user
                    all_sessions = Session.objects.filter(expire_date__gte=datetime.now())
                    if all_sessions.exists():
                        for session in all_sessions:
                            session_data = session.get_decoded()
                            if user.id == int(session_data.get('_auth_user_id')):
                                session.delete()

                    token.delete()
                    return Response({'message': 'logout_succeed'},
                                    status=status.HTTP_200_OK)
                else:
                    return Response({'error': 'token_invalid'},
                                    status=status.HTTP_400_BAD_REQUEST)

            except:
                return Response({'error': 'no_token_sent'},
                                status=status.HTTP_409_CONFLICT)


