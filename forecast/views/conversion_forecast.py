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
            stock_table = stock_data.file_name
            
            predictions_table = scenario.predictions_table_name
            max_historical_date = scenario.max_historical_date
            
            query_predictions = f"SELECT * FROM {predictions_table} WHERE model != 'actual'"
            predictions = pd.read_sql_query(sql=query_predictions, con=engine)

            # Identificar las columnas de fecha desde max_historical_date hasta la última fecha
            date_columns = [col for col in predictions.columns if '-' in col and len(col.split('-')) == 3]
            index = date_columns.index(str(max_historical_date))
            selected_date_columns = date_columns[index:]
            date_columns_sql = ", ".join([f"ROUND(A.`{col}` * B.`Cost Price`, 2) AS `{col}`" for col in selected_date_columns])
            
            if group_by_category is None or group_by_category == "SKU":
                query_predictions = f"""
                    SELECT 
                        A.Family AS Familia,
                        A.Region AS Region,
                        A.Salesman AS Vendedor,
                        A.Client AS Cliente,
                        A.Category AS Categoria,
                        A.Subcategory AS Subcategoria,
                        A.SKU,
                        A.Description AS Descripcion,
                        A.model AS Modelo, 
                        ROUND(B.`Cost Price`, 2) AS "Cost Price", 
                        {date_columns_sql}
                        FROM {predictions_table} A JOIN {stock_table} B ON
                        A.Family = B.Family AND
                        A.Region = B.Region AND
                        A.Salesman = B.Salesman AND
                        A.Client = B.Client AND
                        A.Category = B.Category AND
                        A.Subcategory = B.Subcategory AND
                        A.SKU = B.SKU AND
                        A.Description = B.Description AND
                        A.model != "actual";
                """

                predictions = pd.read_sql_query(sql=query_predictions, con=engine)

                # Convertir el DataFrame en una lista de diccionarios
                result = predictions.to_dict(orient='records')
            
            else:
                
                result = [{"": ""}]


            return Response(result, status=status.HTTP_200_OK)
        
        except ForecastScenario.DoesNotExist:
            return Response({"error": "Scenario not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
