from rest_framework.decorators import authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from ..serializer import FilterData
from ..models import ForecastScenario
from django.db import connection
from ..Graphic import Graphic
import pandas as pd
from files.file_model import FileRefModel
from database.db_engine import engine


class FilterDataViews(APIView):
    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        filters = FilterData(data=request.data)

        if filters.is_valid():
            try:
                scenario_id = filters.validated_data['scenario_id']
                filter_name = filters.validated_data['filter_name']
                filter_value = filters.validated_data['filter_value']
                scenario = ForecastScenario.objects.filter(pk=scenario_id).first()
                error_method = scenario.error_type
                table_name = scenario.predictions_table_name
                pred_p = scenario.pred_p

                with connection.cursor() as cursor:
                    cursor.execute(f'''
                        SELECT * FROM {table_name} 
                        WHERE {filter_name} = "{filter_value}"'''
                    )
                    data_rows = cursor.fetchall()

                    cursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'dtafio' AND TABLE_NAME = '{table_name}';")
                    columns = cursor.fetchall()

                    df_pred = pd.DataFrame(data=data_rows, columns=columns)
                    df_pred = df_pred.drop(columns=[(error_method,)])
                    actual_rows = df_pred[df_pred[('model',)] == 'actual']
                    other_rows = df_pred[df_pred[('model',)] != 'actual']

                    date_columns = [str(col[0]) for col in df_pred.columns[9:]]

                    actual_sum = actual_rows[df_pred.columns[9:]].sum()

                    other_sum = other_rows[df_pred.columns[9:]].sum()

                    actual_data = {'x': date_columns, 'y': actual_sum.tolist()}
                    other_data = {'x': date_columns, 'y': other_sum.tolist()}

                    dates = actual_data['x'][:-pred_p]
                    values = actual_data['y'][:-pred_p]

                    actual_data['x'] = dates
                    actual_data['y'] = values

                    final_data = {'actual_data': actual_data, 'other_data': other_data}
                    data_per_year = Graphic.graphic_predictions_per_year(final_data, max_date=scenario.max_historical_date)

                    return Response({"full_data": final_data, "year_data": data_per_year},
                                    status=status.HTTP_200_OK)
            except Exception as err:
                print(err)

        else:
            return Response({'error': 'bad_request', 'logs': filters.errors}, status=status.HTTP_400_BAD_REQUEST)


class GetFiltersView(APIView):

    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        filters = FilterData(data=request.data)

        if filters.is_valid():
            try:
                scenario_id = filters.validated_data['scenario_id']
                filter_name = filters.validated_data['filter_name']
                scenario = ForecastScenario.objects.filter(pk=scenario_id).first()
                table_name = scenario.predictions_table_name

                if filter_name == 'date':
                    with connection.cursor() as cursor:
                        cursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'dtafio' AND TABLE_NAME = '{table_name}';")
                        columns = cursor.fetchall()

                    date_columns = [x[0] for x in columns if len(x) == 1 and x[0].count('-') == 2]
                    date_columns_str = [str(x).split()[0] if date_columns.index(x) == 0 else str(x) for x in date_columns]
                    
                    return Response(date_columns_str, status=status.HTTP_200_OK)

                else:
                    with connection.cursor() as cursor:
                        cursor.execute(f'''
                        SELECT 
                            {'CONCAT(SKU, " ", DESCRIPTION)' if filter_name == "SKU" else f'DISTINCT({filter_name})'} 
                        FROM {table_name}''')

                        rows = cursor.fetchall()
                        filter_names = []

                        for row in rows:
                            filter_names.append(row[0])

                        return Response(filter_names, status=status.HTTP_200_OK)
            except Exception as err:
                print(err)

        else:
            return Response({'error': 'bad_request', 'logs': filters.errors}, status=status.HTTP_400_BAD_REQUEST)


class FiltersByGroup(APIView):

    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        scenario_id = request.data.get('scenario_id')
        scenario = ForecastScenario.objects.filter(pk=scenario_id).first()
        table = scenario.predictions_table_name
        pred_p = scenario.pred_p

        group = request.data.get('group')
        actual_or_predicted = request.data.get('actual_or_predicted')

        columns_to_delete_hsd = ['family', 'region', 'salesman', 'client', 'category', 'sku', 'description',
                                 'subcategory', 'model']

        try:

            with connection.cursor() as cursor:
                cursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'dtafio' AND TABLE_NAME = '{table}';")
                columns = cursor.fetchall()
                columns_date = []

                for col in columns:
                    if col[0] not in columns_to_delete_hsd:
                        columns_date.append(col[0])

                columns_date = columns_date[:-pred_p] if actual_or_predicted == "actual" else columns_date
                sum_columns = ', '.join([f'SUM(`{date}`)' for date in columns_date])

                if actual_or_predicted == 'both':
                    pass

                else:
                    query = f'''SELECT {group}, {sum_columns} FROM {table}
                                    WHERE model {'=' if actual_or_predicted == 'actual' else '!='} "actual" 
                                    GROUP BY {group} '''

                    cursor.execute(query)
                    data = cursor.fetchall()

                    final_data = {
                        'x': columns_date,
                        'y': {}
                    }

                    for item in data:
                        category_name = item[0]
                        sales_values = item[1:-1]
                        final_data['y'][category_name] = sales_values

            return Response(data=final_data, status=status.HTTP_200_OK)

        except Exception as err:
            return Response({'error': str(err)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FiltersNested(APIView):

    @staticmethod
    def get_historical_table(project: int):
        historical_data = FileRefModel.objects.filter(project_id=project, model_type_id=1).first()
        if historical_data is None:
            raise ValueError("Historical_data_not_found")

        hsd_table = historical_data.file_name

        return hsd_table

    def post(self, request):
        groups_selected = request.data.get('groups')
        filter_to_get = request.data.get('filter_name')
        project_id = request.data.get('project_id')

        try:
            table = self.get_historical_table(project=project_id)

            conditions = [
                f"{filter_name} = '{filter_value}'"
                for filter_dict in groups_selected
                for filter_name, filter_value in filter_dict.items()
            ]

            query = f'SELECT DISTINCT({filter_to_get}) FROM {table} WHERE ' + 'AND ' .join(conditions)

            with connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                filter_names = []

                for row in rows:
                    filter_names.append(row[0])

            return Response(data=filter_names, status=status.HTTP_200_OK)

        except ValueError as err:
            if str(err) == 'Historical_data_not_found':
                return Response(data={'error': 'hsd_not_found'}, status=status.HTTP_400_BAD_REQUEST)