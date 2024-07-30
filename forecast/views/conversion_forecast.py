from rest_framework.decorators import authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from files.file_model import FileRefModel
from ..models import ForecastScenario
from database.db_engine import engine
from django.db import connection
import pandas as pd
import numpy as np
import math
from scipy.special import erfinv
from datetime import datetime, timedelta
from collections import defaultdict
import traceback
import locale

locale.setlocale(locale.LC_ALL, 'es_ES.utf8')

class ConversionForecast(APIView):
    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def get(self, request):
        scenario_id = request.GET.get('scid', None)
        type_of_conversion = request.GET.get('type_of_conversion', None)
        group_by_category = request.GET.get('group_by', None)
        
        try:
            scenario = ForecastScenario.objects.get(pk=scenario_id)
            
            stock_data = FileRefModel.objects.filter(project_id=scenario.project, model_type_id=4).first()
            query_stock = f"SELECT Family, Region, Salesman, Client, Category, Subcategory, SKU, Description, `Cost Price` FROM {stock_data.file_name}"
            stock = pd.read_sql_query(sql=query_stock, con=engine)
            
            predictions_table = scenario.predictions_table_name
            max_historical_date = scenario.max_historical_date
            
            query_predictions = f"SELECT * FROM {predictions_table} WHERE model != 'actual'"
            predictions = pd.read_sql_query(sql=query_predictions, con=engine)
            
            # Convertir los valores 0.0 a "null" en las columnas clave de predictions
            key_columns = ["Family", "Region", "Salesman", "Client", "Category", "Subcategory", "SKU", "Description"]
            for col in key_columns:
                predictions[col] = predictions[col].apply(lambda x: "null" if x == 0.0 else x)
            
            # Identificar las columnas de fecha desde max_historical_date hasta la última fecha
            date_columns = [col for col in predictions.columns if '-' in col and len(col.split('-')) == 3]
            index = date_columns.index(str(max_historical_date))
            selected_date_columns = date_columns[index:]
            
            # Seleccionar las columnas necesarias
            data_cols = key_columns + selected_date_columns
            
            # Filtrar las columnas necesarias de predictions
            processed_predictions = predictions[data_cols]
            
            # Realizar el merge entre stock y processed_predictions utilizando las columnas clave
            merged_data = processed_predictions.merge(stock, on=key_columns, how='left')
            
            # Convertir las columnas de fechas y `Cost Price` a tipos numéricos
            for date_col in selected_date_columns:
                merged_data[date_col] = pd.to_numeric(merged_data[date_col], errors='coerce').fillna(0)
            merged_data['Cost Price'] = pd.to_numeric(merged_data['Cost Price'], errors='coerce').fillna(0)
            
            # Multiplicar las columnas de fechas por el precio de costo correspondiente
            for date_col in selected_date_columns:
                merged_data[date_col] = merged_data[date_col] * merged_data['Cost Price']
            
            # Agrupar por una categoría si se especifica
            if group_by_category:
                grouped_data = merged_data.groupby(group_by_category)[selected_date_columns].sum().reset_index()
            else:
                grouped_data = merged_data[key_columns + selected_date_columns]
            
            # Formatear los números con separadores de miles
            for date_col in selected_date_columns:
                grouped_data[date_col] = grouped_data[date_col].apply(lambda x: locale.format_string("%d", x, grouping=True))
            
            # Convertir a formato de diccionario
            result = grouped_data.to_dict(orient='records')
            
            # Convertir los valores 0.0 a "null"
            for record in result:
                for key, value in record.items():
                    if value == 0.0 and key in key_columns:
                        record[key] = "null"
            
            return Response(result, status=status.HTTP_200_OK)
        
        except ForecastScenario.DoesNotExist:
            return Response({"error": "Scenario not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)