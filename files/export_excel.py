from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import authentication_classes, permission_classes
from .serializer import ExcelToExportSerializer
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from projects.models import ProjectsModel
from django.conf import settings
import pandas as pd
import os


class ExportExcelAPIView(APIView):
    @staticmethod
    def create_excel(rows: list, columns: list):
        try:
            dataframe = pd.DataFrame(rows, columns=columns)
            return dataframe

        except Exception as e:
            print((str(e)))
            raise ValueError(f"Error: {str(e)}")

    @staticmethod
    def create_export_directory():
        directory = os.path.join(settings.MEDIA_ROOT, 'excel_files/exported_files/')
        if not os.path.exists(directory):
            os.makedirs(directory)


    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        serializer = ExcelToExportSerializer(data=request.data)

        try:
            if serializer.is_valid():
                project = ProjectsModel.objects.filter(id=serializer.validated_data['project_pk']).first()
                try:
                    dataframe = self.create_excel(rows=serializer.validated_data['rows'],
                                                  columns=serializer.validated_data['columns'])

                except ValueError as ve:
                    return Response({'error': 'error_creating_dataframe', 'logs': str(ve)},
                                    status=status.HTTP_400_BAD_REQUEST)

                self.create_export_directory()
                file_name = serializer.validated_data['file_name']
                file_path = os.path.join('media', 'excel_files', 'exported_files', f'{file_name}_project_{project.project_name}.xlsx')

                try:
                    with pd.ExcelWriter(file_path, engine='xlsxwriter') as excel_writer:
                        dataframe.to_excel(excel_writer, sheet_name='Result', index=True, merge_cells=False)

                    
                    return Response({'file_url': file_path}, status=status.HTTP_200_OK)

                except Exception as e:
                    print(f"ERROR {str(e)}")
                    return Response({'error': 'error_saving_excel', 'logs': str(e)},
                                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            else:
                return Response({'error': 'bad_request', 'logs': serializer.errors},
                                status=status.HTTP_400_BAD_REQUEST)

        except Exception as err:
            print(str(err))
            return Response({'error': 'internal_server_error', 'logs': str(err)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
