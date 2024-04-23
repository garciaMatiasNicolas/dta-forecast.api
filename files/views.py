from .file_model import FileRefModel, FileTypes
from .filemanager import save_dataframe
from .serializer import FileSerializer, FileModelType
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import authentication_classes, permission_classes, action
from rest_framework import viewsets
from rest_framework.parsers import FormParser, MultiPartParser
from django.db import connection, OperationalError
import os


@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
class ExcelFileUploadView(viewsets.ModelViewSet):
    queryset = FileRefModel.objects.all()
    serializer_class = FileSerializer
    parser_classes = (MultiPartParser, FormParser)

    def list(self, request, *args, **kwargs):
        user_id = request.user.id
        files = self.get_queryset().filter(user_owner_id=user_id)
        serializer = self.get_serializer(files, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def create(self, request, *args, **kwargs):
        file_serializer = self.get_serializer(data=request.data)
        file_model = file_serializer.initial_data['model_type']

        if file_serializer.is_valid():
            # Get validated data from request
            file_name = file_serializer.validated_data['file_name']
            project = file_serializer.validated_data['project']
            model_type = FileTypes.objects.get(id=file_model)

            # Save file
            file_serializer.save()
            file_ref_model = file_serializer.instance

            # Get file route and instance DataFrame
            route = str(file_serializer.data['file'])

            # Save dataframe
            try:
                save_dataframe(route_file=route, model_type=model_type.model_type, file_name=file_name, wasSaved=False,
                               project_pk=project)

                return Response({'message': 'file_uploaded'},
                                status=status.HTTP_201_CREATED)

            except ValueError as err:
                err = str(err)
                print(err)
                # Delete file from server
                route = os.path.join('media', route)
                if os.path.exists(route):
                    os.remove(route)

                # Delete file from db
                pk_file = file_ref_model.id
                file = self.get_queryset().filter(id=pk_file)
                file.delete()

                if err == "model_not_allowed":
                    return Response(data={'error': 'model_not_allowed'}, status=status.HTTP_400_BAD_REQUEST)

                if err == "columns_not_in_date_type":
                    return Response(data={'error': 'columns_not_in_date_type'},
                                    status=status.HTTP_400_BAD_REQUEST)

                if err == "cols_exog_endog_not_match":
                    return Response(data={'error': 'dates_dont_match'}, status=status.HTTP_400_BAD_REQUEST)

                return Response({'error': 'other_value_error'}, status=status.HTTP_400_BAD_REQUEST)

        else:
            return Response({'error': 'bad_request', 'logs': file_serializer.errors},
                            status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk = None):
        file_to_destroy = self.get_queryset().filter(id=pk).first()

        if file_to_destroy:
            file_url = os.path.join('media', str(file_to_destroy.file))

            if os.path.exists(file_url):
                os.remove(file_url)

            try:
                with connection.cursor() as cursor:
                    cursor.execute(f'DROP TABLE {file_to_destroy.file_name}')
            except OperationalError:
                pass

            file_to_destroy.delete()
            return Response({'message': 'file_deleted'}, status=status.HTTP_200_OK)

        else:
            return Response({'error': 'file_not_found'}, status=status.HTTP_400_BAD_REQUEST)


@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
class GetFileTypes(APIView):
    def post(self, request):
        serializer = FileModelType(data=request.data)

        if serializer.is_valid():
            serializer.save()
            return Response({'message': 'created'})

    def get(self, request):
        files = FileTypes.objects.all()
        serializer = FileModelType(files, many=True)
        return Response(serializer.data)

