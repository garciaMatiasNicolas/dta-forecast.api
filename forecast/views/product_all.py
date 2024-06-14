from rest_framework.decorators import authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from ..models import ForecastScenario
from database.db_engine import engine
from django.db import connection
import pandas as pd
from files.file_model import FileRefModel
from .report_data_view import ReportDataViews

class AllProductView(APIView):
    
    @staticmethod
    def get_data(project_pk: int, product: dict, scenario_pk: int = None):
        # Limpiar los valores de texto en el diccionario 'product'
        cleaned_product = {key: value.strip() if isinstance(value, str) else value for key, value in product.items()}

        # Construir la parte de la consulta WHERE de forma dinámica
        conditions = " AND ".join([f"{key} = '{value}'" if isinstance(value, str) else f"{key} = {value}" for key, value in cleaned_product.items()])

        if scenario_pk is not None:
            table = ForecastScenario.objects.get(pk=scenario_pk)
            query = f"SELECT * FROM {table.predictions_table_name} WHERE {conditions};"
            print(query)
        else:
            table = FileRefModel.objects.filter(project_id=project_pk, model_type_id=1).first()
            query = f"SELECT * FROM {table.file_name} WHERE {conditions};"

        data = pd.read_sql_query(query, engine)
        return data
    
    @staticmethod
    def calculate_kpis(predictions_table_name, last_date_index, list_date_columns, product):
        try:
            with (connection.cursor() as cursor):
                last_year_since_last_date = list_date_columns[last_date_index - 12:last_date_index + 1][1:]
                last_quarter_since_last_date = list_date_columns[last_date_index - 3:last_date_index + 1]
                last_month = list_date_columns[last_date_index]

                next_year_since_last_date = list_date_columns[last_date_index + 1:last_date_index + 13]
                next_quarter_since_last_date = list_date_columns[last_date_index + 1:last_date_index + 5]
                next_month_since_last_date = list_date_columns[last_date_index + 1:last_date_index + 2]

                if last_date_index - 23 >= 0:
                    dates_a = list_date_columns[last_date_index - 23:last_date_index - 11]
                else:
                    dates_a = list_date_columns[:last_date_index - 11]
                    
                dates_b = dates_a[-4:]
                dates_c = dates_a[-1]

                dates_d = last_year_since_last_date[:4]
                dates_e = dates_d[0]

                reports_name = [
                    "last_year_since_last_date",
                    "last_quarter_since_last_date",
                    "last_month",
                    "next_year_since_last_date",
                    "next_quarter_since_last_date",
                    "next_month_since_last_date",
                    "dates_a",
                    "dates_b",
                    "dates_c",
                    "dates_d",
                    "dates_e"
                ]

                date_ranges = [
                    last_year_since_last_date,
                    last_quarter_since_last_date,
                    last_month,
                    next_year_since_last_date,
                    next_quarter_since_last_date,
                    next_month_since_last_date,
                    dates_a,
                    dates_b,
                    dates_c,
                    dates_d,
                    dates_e
                ]

                reports_data = {}

                for date_range, date_name in zip(date_ranges, reports_name):
                    dates_report = ReportDataViews.join_dates(list_dates=date_range, for_report=True)
                    reports_data[date_name] = dates_report
                

                actual_dates = f'''
                    SELECT
                        SKU || " " || DESCRIPTION,
                        ROUND({reports_data["last_year_since_last_date"]}),
                        ROUND({reports_data["last_quarter_since_last_date"]}),
                        ROUND(SUM(`{last_month}`)),
                        ROUND({reports_data["dates_a"]}),
                        ROUND({reports_data["dates_b"]}),
                        ROUND(SUM(`{dates_c}`)),
                        ROUND({reports_data["dates_d"]}),
                        ROUND(SUM(`{dates_e}`))
                    FROM {predictions_table_name}
                    WHERE model = 'actual' AND SKU = "{product}"
                    GROUP BY SKU, DESCRIPTION;
                '''

                predicted_dates = f'''
                    SELECT
                        SKU || " " || DESCRIPTION,
                        ROUND({reports_data["next_year_since_last_date"]}),
                        ROUND({reports_data["next_quarter_since_last_date"]}),
                        ROUND({reports_data["next_month_since_last_date"]})
                    FROM {predictions_table_name}
                    WHERE model != 'actual' AND SKU = "{product}"
                    GROUP BY SKU, DESCRIPTION;
                '''

                cursor.execute(sql=actual_dates)
                actual_dates = cursor.fetchall()

                cursor.execute(sql=predicted_dates)
                predicted_dates = cursor.fetchall()
            
                final_data = []

                for predicted, actual in zip(predicted_dates, actual_dates):
                    # Verificar si las categorías coinciden
                    if predicted[0] == actual[0]:
                        # Calcular los porcentajes
                        ytd = ReportDataViews.calc_perc(n1=actual[1], n2=actual[4])
                        qtd = ReportDataViews.calc_perc(n1=actual[2], n2=actual[5])
                        mtd = ReportDataViews.calc_perc(n1=actual[3], n2=actual[6])
                        ytg = ReportDataViews.calc_perc(n1=predicted[1], n2=actual[1])
                        qtg = ReportDataViews.calc_perc(n1=predicted[2], n2=actual[7])
                        mtg = ReportDataViews.calc_perc(n1=predicted[3], n2=actual[8])
                        
                        # Agregar los resultados a la lista final
                        final_data.append([ytd, qtd, mtd, ytg, qtg, mtg])

                # Retornar los datos finales
                return final_data
            
        except Exception as err:
            print(err)


    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        scenario = request.data.get('scenario_pk')
        product = request.data.get('product')  # Asegúrate de que 'product' es un diccionario con los campos correctos
        project = request.data.get('project_pk')

        data = self.get_data(project_pk=project, product=product, scenario_pk=scenario)

        if scenario is not None:
            scenario_obj = ForecastScenario.objects.get(pk=scenario)
            error_val = data[scenario_obj.error_type]
            max_date = scenario_obj.max_historical_date
            date_columns = [col for col in data.columns if '-' in col and len(col.split('-')) == 3]
            index = date_columns.index(str(max_date))
            values = data[date_columns].values.tolist()
            kpis = self.calculate_kpis(predictions_table_name=scenario_obj.predictions_table_name, last_date_index=index, list_date_columns=date_columns, product=product["SKU"])

            final_data = {
                "product": f"{product['SKU']}",
                "graphic_forecast": {"dates": date_columns, "values": values[1]},
                "graphic_historical": {"dates": date_columns, "values": values[0]},
                "error": max(error_val),
                "kpis": {"columns": ["YTD", "QTD", "MTD", "YTG", "QTG", "MTG"], "values": kpis[0]},
            }

            return Response(final_data, status=status.HTTP_200_OK)
        
        return Response({"error": "Scenario not found"}, status=status.HTTP_400_BAD_REQUEST)

