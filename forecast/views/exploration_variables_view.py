from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.decorators import authentication_classes, permission_classes, action
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from database.db_engine import engine
from files.file_model import FileRefModel
from django.db import connection
import pandas as pd
import locale
locale.setlocale(locale.LC_ALL, 'es_ES.utf8')

class FilterValuesView(APIView):
    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        project_pk = request.data.get('project_id')
        filter_name = request.data.get('filter_name')

        hsd = FileRefModel.objects.filter(project_id=project_pk, model_type_id=1).first()

        with connection.cursor() as cursor:
            cursor.execute(f'SELECT DISTINCT({filter_name}) FROM {hsd.file_name}')
            rows = cursor.fetchall()
            filter_names = []

            for row in rows:
                filter_names.append(row[0])

        return Response(filter_names, status=status.HTTP_200_OK)


class HistoricalDataView(APIView):

    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        project_pk = request.data.get('project_id')
        filter_name = request.data.get('filter_name')
        component = request.data.get('component')

        hsd = FileRefModel.objects.filter(project_id=project_pk, model_type_id=1).first()
        columns_to_delete_hsd = ['Family', 'Region', 'Salesman', 'Client', 'Category', 'SKU', 'Description',
                                 'Subcategory', 'Starting Year', 'Starting Period',
                                 'Periods Per Year', 'Periods Per Cycle']
        try:
            if component == 'graph':

                if filter_name == 'all':
                    hsd_table = pd.read_sql_table(table_name=hsd.file_name, con=engine)
                    date_columns_hsd = hsd_table.columns[12:]
                    hsd_sum = hsd_table[date_columns_hsd].sum()
                    hsd_data = {'x': date_columns_hsd.to_list(), 'y': hsd_sum.to_list()}
                    return Response(data=hsd_data, status=status.HTTP_200_OK)

                else:
                    with connection.cursor() as cursor:
                        cursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'dtafio' AND TABLE_NAME = '{hsd.file_name}';")
                        columns = cursor.fetchall()
                        columns_date = []

                        for col in columns:
                            if col[0] not in columns_to_delete_hsd:
                                columns_date.append(col[0])

                        sum_columns = ', '.join([f'SUM(`{date}`)' for date in columns_date])

                        cursor.execute(f'SELECT {filter_name}, {sum_columns} FROM {hsd.file_name} GROUP BY {filter_name}')
                        data = cursor.fetchall()

                data_dict = {
                    'x': columns_date,
                    'y': {}
                }

                for item in data:
                    category_name = item[0]
                    sales_values = item[1:]
                    data_dict['y'][category_name] = sales_values
                

                return Response(data=data_dict, status=status.HTTP_200_OK)
            
            else:
                hsd_table = pd.read_sql_table(table_name=hsd.file_name, con=engine)
                data_table = hsd_table.to_dict(orient='records')
                date_columns_hsd = hsd_table.columns[12:]

                for row in data_table:
                    for col in date_columns_hsd:
                        if pd.notnull(row[col]):  # Verificar si el valor no es nulo
                            row[col] = locale.format_string("%d", int(round(row[col])), grouping=True)

                return Response(data=data_table, status=status.HTTP_200_OK)
                
        except Exception as err:
            print(err)

    


class AllocationMatrixView(APIView):
    @staticmethod
    def calculate_allocation_matrix(var_name, row_exog_data, row_historical_data):
        correlation = row_exog_data.corr(row_historical_data)
        return {var_name: str(round(correlation.mean(), 3))}

    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        project_pk = request.data.get('project_id')
        group = request.data.get('group')
        group_val = request.data.get('group_val')

        hsd = FileRefModel.objects.filter(project_id=project_pk, model_type_id=1).first()
        exog = FileRefModel.objects.filter(project_id=project_pk, model_type_id=2).first()

        if exog is None:
            return Response(data={'error': 'not_exog_data'}, status=status.HTTP_400_BAD_REQUEST)

        historical_data_table = hsd.file_name
        exog_data_table = exog.file_name

        try:

            df_historical_exogenous = pd.read_sql_table(table_name=exog_data_table, con=engine)
            df_historical_data = pd.read_sql_table(table_name=historical_data_table, con=engine)

            if group is not None and group_val is not None:
                df_historical_data = df_historical_data[df_historical_data[group] == group_val]

            data = {}

            for _, row_historical in df_historical_data.iterrows():
                product_name = ' || '.join(str(row_historical[col]) for col in df_historical_data.iloc[:, :8].columns if
                                           str(row_historical[col]) != 'null')
                correlations = []

                for _, row_exog in df_historical_exogenous.iterrows():
                    if (
                            (row_exog['Family'] == 'all_data') or
                            (row_exog['Family'] == row_historical['Family']) or
                            (row_exog['Region'] == row_historical['Region']) or
                            (row_exog['Category'] == row_historical['Category']) or
                            (row_exog['Subcategory'] == row_historical['Subcategory']) or
                            (row_exog['Client'] == row_historical['Client']) or
                            (row_exog['Salesman'] == row_historical['Salesman'])
                    ):
                        corr_dict = self.calculate_allocation_matrix(
                            var_name=row_exog['Variable'],
                            row_exog_data=row_exog[8:],
                            row_historical_data=row_historical[12:]
                        )
                        correlations.append(corr_dict)

                if product_name not in data:
                    data[product_name] = correlations

            return Response(data=data, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response(data={"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GetExogVars(APIView):
    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        project_pk = request.data.get('project_id')
        exog = FileRefModel.objects.filter(project_id=project_pk, model_type_id=2).first()

        if exog is None:
            return Response({'error': 'not_exog_data'}, status=status.HTTP_400_BAD_REQUEST)

        list_of_variables = []

        with connection.cursor() as cursor:
            query = f'SELECT DISTINCT(Variable) FROM {exog.file_name}'
            cursor.execute(query)
            data = cursor.fetchall()
            for i in data:
                list_of_variables.append(i[0])

        return Response(list_of_variables, status=status.HTTP_200_OK)


class ExogenousVariablesTable(APIView):
    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        project_id = request.data.get("project_id")
        group = request.data.get('group')
        hev = FileRefModel.objects.filter(project_id=project_id, model_type_id=2).first()

        if hev is None:
            return Response(data={'error': 'not_data'}, status=status.HTTP_400_BAD_REQUEST)

        hev_table = pd.read_sql_table(table_name=hev.file_name, con=engine)
        date_columns = hev_table.columns[8:].to_list()
        rows = []

        if group is None:
            for col, row in hev_table.iterrows():
                rows.append(row.to_list())

        else:
            with connection.cursor() as cursor:
                date_columns_str = ', '.join([f'"{date}"' for date in date_columns])

                cursor.execute(f'''SELECT  VARIABLE, {group},
                                           {date_columns_str}  
                                    FROM {hev.file_name} WHERE {group} != "nan" ''')

                filtered_table = cursor.fetchall()

                for row in filtered_table:
                    rows.append(row)

        data = {
            'date_columns': date_columns,
            'rows': rows
        }

        return Response(data=data, status=status.HTTP_200_OK)


class ExogenousVariablesGraph(APIView):
    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        project_id = request.data.get("project_id")
        hev = FileRefModel.objects.filter(project_id=project_id, model_type_id=2).first()

        if hev is None:
            return Response(data={'error': 'not_data'}, status=status.HTTP_400_BAD_REQUEST)

        hev_table = pd.read_sql_table(table_name=hev.file_name, con=engine)
        date_columns = hev_table.columns[8:].to_list()
        data_for_each_date = []

        with connection.cursor() as cursor:
            date_columns_str = ', '.join([f'"{date}"' for date in date_columns])

            cursor.execute(f'''SELECT Variable {date_columns_str} FROM {hev.file_name} ''')
            table = cursor.fetchall()

            for row in table:
                data = list(row)
                data_for_each_date.append(data)

        data = {
            'data': data_for_each_date,
            'dates': date_columns
        }

        return Response(data=data, status=status.HTTP_200_OK)
