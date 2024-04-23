from rest_framework.decorators import authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework import status
from ..serializer import ForecastScenarioSerializer
from ..models import ForecastScenario
from django.db import connection, OperationalError
import os


@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
class ForecastScenarioViewSet(ModelViewSet):
    serializer_class = ForecastScenarioSerializer
    queryset = ForecastScenario.objects.all()

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset().filter(user_id=request.user.id)
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def create(self, request, *args, **kwargs):
        scenario = self.get_serializer(data=request.data)

        if scenario.is_valid():
            project_id = scenario.validated_data['project']
            scenario_name = scenario.validated_data['scenario_name']

            search_scenario_name = ForecastScenario.objects.filter(project_id=project_id, scenario_name=scenario_name)

            if search_scenario_name.exists():
                return Response({'error': 'scenario_name_already_exists'}, status=status.HTTP_400_BAD_REQUEST)

            scenario = scenario.save()

            return Response({'message': 'scenario_saved', 'scenario_id': scenario.id},
                            status=status.HTTP_201_CREATED)

        else:
            return Response({'error': 'bad_request', 'logs': scenario.errors},
                            status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None):
        scenario_to_update = self.get_queryset().filter(id=pk).first()
        scenario_name = request.data.get("scenario_name")

        if scenario_to_update:
            scenario_to_update.scenario_name = scenario_name
            scenario_to_update.save()
            return Response({'message': 'scenario_name_updated'}, status=status.HTTP_200_OK)

        else:
            return Response({'error': 'scenario_not_found'}, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        scenario_to_destroy = self.get_queryset().filter(id=pk).first()

        if scenario_to_destroy:
            table_name = scenario_to_destroy.predictions_table_name
            excel_with_predictions = scenario_to_destroy.url_predictions if scenario_to_destroy.url_predictions is not None else ""

            try:
                if os.path.exists(excel_with_predictions):
                    os.remove(excel_with_predictions)

                try:
                    with connection.cursor() as cursor:
                        cursor.execute(f'DROP TABLE {table_name}')

                except OperationalError as dbError:
                    pass

                scenario_to_destroy.delete()
                return Response({'message': 'scenario_deleted'}, status=status.HTTP_200_OK)

            except Exception as error:
                print(error)
                return Response({'error': 'server_error'}, status=status.HTTP_400_BAD_REQUEST)

        else:
            return Response({'error': 'scenario_not_found'},
                            status=status.HTTP_400_BAD_REQUEST)