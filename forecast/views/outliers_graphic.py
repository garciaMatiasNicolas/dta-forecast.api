from rest_framework.views import APIView
from rest_framework.decorators import permission_classes, authentication_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from rest_framework.response import Response
from files.file_model import FileRefModel
from database.db_engine import engine
from django.db import connection
from scipy import stats
import pandas as pd
import numpy as np


class GraphicOutliersView(APIView):

    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        sku = request.data.get("sku")
        threshold = request.data.get("threshold")
        project_id = request.data.get("project")
        filter_group = request.data.get("filter_group")
        filter_val = request.data.get("filter_val")

        hsd = FileRefModel.objects.filter(project_id=project_id, model_type_id=1).first()

        if hsd is None:
            return Response(data={'error': 'not_data'}, status=status.HTTP_400_BAD_REQUEST)

        hsd_table = pd.read_sql_table(table_name=hsd.file_name, con=engine)

        if sku is not None:
            row_with_sku = hsd_table.loc[hsd_table['SKU'] == sku]

            if row_with_sku.empty:
                return Response(data={'error': 'sku_not_found'}, status=status.HTTP_404_NOT_FOUND)

            date_columns_hsd = row_with_sku.columns[12:].to_list()
            sales_for_each_date = row_with_sku[date_columns_hsd].sum().to_list()

            outlier_df = pd.DataFrame({
                "Dates": pd.to_datetime(date_columns_hsd),
                "Sales": sales_for_each_date
            })

        else:
            if filter_val is not None and filter_group is not None:
                hsd_filtered = hsd_table.loc[hsd_table[filter_group] == filter_val]
                date_columns_hsd = hsd_filtered.columns[12:].to_list()
                sales_for_each_date = hsd_filtered[date_columns_hsd].sum().to_list()

                outlier_df = pd.DataFrame({
                    "Dates": pd.to_datetime(date_columns_hsd),
                    "Sales": sales_for_each_date
                })

            else:
                date_columns_hsd = hsd_table.columns[12:].to_list()
                sales_for_each_date = hsd_table[date_columns_hsd].sum().to_list()

                outlier_df = pd.DataFrame({
                    "Dates": pd.to_datetime(date_columns_hsd),
                    "Sales": sales_for_each_date
                })

        # Calculate the z-score for each sale
        z = np.abs(stats.zscore(outlier_df['Sales']))

        # Identify outliers as sales with a z-score greater than 2
        outliers = outlier_df[z > threshold]

        dates_with_outliers = [str(date.date()) for date in outliers['Dates']]

        columns_to_keep = ['Family', 'Region', 'Salesman', 'Client', 'Category', 'Subcategory', 'SKU', 'Description']
        columns_to_keep.extend(dates_with_outliers)

        if filter_val is not None and filter_group is not None:
            table_report = hsd_filtered[columns_to_keep]

        else:
            table_report = hsd_table[columns_to_keep]

        rows = []
        for col, row in table_report.iterrows():
            rows.append(row.to_list())

        if sku is not None:
            table_rows_filtered = [row for row in rows if row[6] == sku]
            data = {
                'dates': date_columns_hsd,
                'sales': sales_for_each_date,
                'outliers': dates_with_outliers,
                'table_rows': table_rows_filtered
            }

        else:
            data = {
                'dates': date_columns_hsd,
                'sales': sales_for_each_date,
                'outliers': dates_with_outliers,
                'table_rows': rows
            }

        return Response(data=data, status=status.HTTP_200_OK)


class FiltersHistoricalData(APIView):
    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        print(request.data)
        project_id = request.data.get("project")
        filter_name = request.data.get("filter_name")

        hsd = FileRefModel.objects.filter(project_id=project_id, model_type_id=1).first()

        with connection.cursor() as cursor:
            cursor.execute(f'SELECT DISTINCT({filter_name}) FROM {hsd.file_name}')
            rows = cursor.fetchall()
            filter_names = []

            for row in rows:
                filter_names.append(row[0])

            return Response(filter_names, status=status.HTTP_200_OK)
