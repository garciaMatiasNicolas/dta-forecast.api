from typing import List, Dict, Any, Tuple
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
from scipy.special import erfinv
from datetime import datetime, timedelta
from collections import defaultdict
import traceback


class StockDataView(APIView):

    # -- METHODS FOR GET DATA FROM DB -- #
    @staticmethod
    def get_data(project_pk: int, scenario: int = None):
        if scenario:
            forecast_data = ForecastScenario.objects.get(pk=scenario)
            
            if forecast_data is None:
                table_forecast = None
        
            else:
                query = f"SELECT * FROM {forecast_data.predictions_table_name} WHERE model != 'actual'"
                table_forecast = pd.read_sql_query(query, engine)
        
        else:
            table_forecast = None

        historical_data = FileRefModel.objects.filter(project_id=project_pk, model_type_id=1).first()
        if historical_data is None:
            raise ValueError("Historical_not_found")

        table_historical = pd.read_sql_table(table_name=historical_data.file_name, con=engine)

        stock_data = FileRefModel.objects.filter(project_id=project_pk, model_type_id=4).first()
        if stock_data is None:
            raise ValueError("Stock_data_not_found")

        table_stock = pd.read_sql_table(table_name=stock_data.file_name, con=engine)

        tables = {"historical": table_historical, "stock": table_stock, "forecast": table_forecast}

        return tables

    # --- METHODS FOR CALCULATE STOCK --- #
    @staticmethod
    def calculate_avg_desv_varcoefficient(historical: pd.DataFrame, stock: pd.DataFrame, forecast_periods: int, forecast):
        results = []
        iterrows_historical = historical.iloc[:, -12:].iterrows()

        historical.fillna(0)

        if forecast is not None:
            iterrows_forecast = forecast.iloc[:, -forecast_periods:].iterrows() 
            forecast.fillna(0) 
            
            for (_, row_historical), ( _, row_forecast) in zip(iterrows_historical, iterrows_forecast):
                total_sales_historical = row_historical.sum()
                total_sales_forecast = row_forecast.sum()

                avg_row_historical = np.average(row_historical)
                avg_row_forecast = np.average(row_forecast)

                avg_sales_historical = round(avg_row_historical / 30, 2)
                avg_sales_forecast = round(avg_row_forecast / 30, 2)

                desv_historical = round(row_historical.std(), 2)
                desv_forecast = round(row_forecast.std(), 2)
                
                coefficient_of_variation_historical = round(desv_historical / avg_sales_historical, 2) if avg_sales_historical != 0 else 0
                coefficient_of_variation_forecast = round(desv_forecast / avg_sales_forecast, 2) if avg_sales_forecast != 0 else 0

                stock_or_request_historical = 'stock' if coefficient_of_variation_historical > 0.7 else 'request'
                stock_or_request_forecast = 'stock' if coefficient_of_variation_forecast > 0.7 else 'request'
                
                desv_2_historical = round(desv_historical / 30, 2)
                desv_2_forecast = round(desv_forecast / 30, 2)

                avg_row_historical = str(avg_row_historical) if not pd.isna(avg_row_historical) else '0.0'
                avg_row_forecast = str(avg_row_forecast) if not pd.isna(avg_row_forecast) else '0.0'

                desv_historical = str(desv_historical) if not pd.isna(desv_historical) else '0.0'
                desv_forecast = str(desv_forecast) if not pd.isna(desv_forecast) else '0.0'

                coefficient_of_variation_historical = str(coefficient_of_variation_historical) if not pd.isna(coefficient_of_variation_historical) else '0.0'
                coefficient_of_variation_forecast = str(coefficient_of_variation_forecast) if not pd.isna(coefficient_of_variation_forecast) else '0.0'

                row_with_stats = {
                    'total_sales_historical': total_sales_historical,
                    'avg_row_historical': avg_row_historical,
                    'desv_historical': desv_historical,
                    'coefficient_of_variation_historical': coefficient_of_variation_historical,
                    'stock_or_request_historical': stock_or_request_historical,
                    'avg_sales_per_day_historical': avg_sales_historical,
                    'desv_per_day_historical': desv_2_historical,
                    'total_sales_forecast': total_sales_forecast,
                    'avg_row_forecast': avg_row_forecast,
                    'desv_forecast': desv_forecast,
                    'coefficient_of_variation_forecast': coefficient_of_variation_forecast,
                    'stock_or_request_forecast': stock_or_request_forecast,
                    'avg_sales_per_day_forecast': avg_sales_forecast,
                    'desv_per_day_forecast': desv_2_forecast
                }

                results.append(row_with_stats)
        
        else:
            for _, row in iterrows_historical:
                total_sales = row.sum()
                avg_row = np.average(row)
                avg_sales = round(avg_row / 30,2)
                desv = round(row.std(), 2)
                coefficient_of_variation = round(desv / avg_sales, 2) if avg_sales != 0 else 0
                stock_or_request = 'stock' if coefficient_of_variation > 0.7 else 'request'
                desv_2 = round(desv / 30, 2)
        
                avg_row = str(avg_row) if not pd.isna(avg_row) else '0.0'
                desv = str(desv) if not pd.isna(desv) else '0.0'
                coefficient_of_variation = str(coefficient_of_variation) if not pd.isna(coefficient_of_variation) else '0.0'

                row_with_stats = {
                   'total_sales_historical': total_sales,
                    'avg_row_historical': avg_row,
                    'avg_row_forecast': "0.0",
                    'desv_historical': desv,
                    'coefficient_of_variation_historical': coefficient_of_variation,
                    'stock_or_request_historical': stock_or_request,
                    'avg_sales_per_day_historical': avg_sales,
                    "avg_sales_per_day_forecast": "0.0",
                    'desv_per_day_historical': desv_2
                }
                
                results.append(row_with_stats)

        stats_df = pd.DataFrame(results)

        result_df = pd.concat(objs=[historical[['SKU', 'Description', 'Family', 'Region', 'Client', 'Salesman', 'Category', 'Subcategory']], stats_df], axis=1)
        result_df = pd.merge(result_df, stock, on=['SKU', 'Description', 'SKU', 'Description', 'Family', 'Region', 'Client', 'Salesman', 'Category', 'Subcategory'])
        result_list = result_df.to_dict(orient='records')

        return result_list

    @staticmethod
    def calculate_abc(products, is_forecast):
        total = sum(product[f'total_sales_{"forecast" if is_forecast else "historical"}'] for product in products)
        products.sort(key=lambda x: x[f'total_sales_{"forecast" if is_forecast else "historical"}'], reverse=True)

        abc_data = []
        acum = 0

        for product in products:
            acum += product[f'total_sales_{"forecast" if is_forecast else "historical"}']
            abc_class = 'A' if acum / total <= 0.2 else (
                'B' if acum / total <= 0.8 else 'C')

            abc = {"SKU": product['SKU'], "ABC": abc_class}
            abc_data.append(abc)

        return abc_data

    def calculate_stock(self, data: List[Dict[str, Any]], next_buy_days: int, is_forecast: bool) -> (
            tuple)[list[dict[str | Any, int | str | datetime | Any]], bool]:

        def verify_safety_stock_zero(array: List[Dict[str, Any]]):
            for product in array:
                if product.get("Safety Stock", 0) != 0:
                    return False

            return True

        safety_stock_is_zero = verify_safety_stock_zero(data)

        results = []

        abc = self.calculate_abc(products=data, is_forecast=is_forecast)
        abc_dict = {product['SKU']: product['SKU'] for product in abc}

        for item in data:
            avg_sales_historical = float(item["avg_sales_per_day_historical"])
            price = float(item['Price'])
            avg_sales_forecast = float(item["avg_sales_per_day_forecast"]) 
            avg_sales = float(item[f'avg_sales_per_day_{"forecast" if is_forecast else "historical"}'])
            available_stock = float(item['Available Stock'])
            lead_time = int(item['Lead Time'])
            safety_stock = int(item['Safety stock'])
            reorder_point = next_buy_days + lead_time + safety_stock
            days_of_coverage = round(available_stock / avg_sales) if avg_sales != 0 else 9999
            buy = 'Si' if (days_of_coverage - reorder_point) < 1 else 'No'
            optimal_batch = float(item["Optimal Batch"])
            how_much = max(optimal_batch, (next_buy_days + lead_time + safety_stock - days_of_coverage) * avg_sales ) if buy == 'Si' else 0
            overflow_units = available_stock if avg_sales == 0 else (0 if days_of_coverage - reorder_point < 0 else round((days_of_coverage - reorder_point)*avg_sales/30)) 
            overflow_price = round(overflow_units*price)
            try:
                next_buy = datetime.now() + timedelta(days=days_of_coverage - lead_time) if days_of_coverage != 0 \
                    else datetime.now()

            except OverflowError:
                next_buy = ""

            if days_of_coverage == 9999:
                stock_status = "Obsoleto"
                if available_stock != 0:
                    characterization = "0-Con stock sin ventas"
            elif days_of_coverage > 360:
                stock_status = 'Alto sobrestock'
                characterization = "1-Más de 360 días"
            elif days_of_coverage > 180:
                stock_status = 'Sobrestock'
                characterization = "2-Entre 180 y 360"
            elif days_of_coverage > 30:
                stock_status = 'Normal'
                if days_of_coverage > 90:
                    characterization = "3-Entre 90 y 180"
                else:
                    characterization = "4-Entre 30 y 90"
            elif days_of_coverage > 15:
                stock_status = 'Riesgo quiebre'
                characterization = "5-Entre 15 y 30"
            elif days_of_coverage >= 0:
                stock_status = "Quiebre"
                characterization = "6-Menos de 15"
            else:
                stock_status = 'Stock negativo'
                characterization = "Sin stock"

            next_buy = next_buy.strftime('%Y-%m-%d') if isinstance(next_buy, datetime) else next_buy

            stock = {
                'Familia': item['Family'],
                'Categoria': item['Category'],
                'Vendedor': item['Salesman'],
                'Subcategoria': item['Subcategory'],
                'Cliente': item['Client'],
                'Región': item['Region'],
                'SKU': str(item['SKU']),
                'Descripción': str(item['Description']),
                'Stock': str(available_stock),
                'Venta diaria histórico': str(avg_sales_historical),
                'Venta diaria predecido': str(avg_sales_forecast),
                'Cobertura (días)': str(days_of_coverage),
                'Stock seguridad en dias': str(safety_stock),
                'Punto de reorden': str(reorder_point),
                '¿Compro?': str(buy),
                '¿Cuanto?': str(round(how_much)),
                'Estado': str(stock_status),
                'Valorizado': f'${str(round(price*available_stock, 2))}',
                'Demora en dias': str(lead_time),
                'Fecha próx. compra':  str(next_buy) if days_of_coverage != 9999 else "---",
                'Caracterización': characterization,
                'Sobrante (unidades)': str(overflow_units),
                'Cobertura prox. compra (días)': str( days_of_coverage- next_buy_days ),
                'Sobrante valorizado': f'${str(overflow_price)}',
                'ABC': abc_dict.get(item['SKU'], ''),
                'XYZ': item['XYZ']
            }

            results.append(stock)

        return results, safety_stock_is_zero

    @staticmethod
    def calculate_safety_stock(data: List[Dict[str, Any]]):
        final_data = []

        for product in data:
            avg_sales_per_day = product['avg_sales_per_day_historical']
            desv_per_day = product['desv_per_day_historical']
            lead_time = product['Lead Time']
            service_level = product['Service Level'] / 100
            desv_est_lt_days = product['Desv Est Lt Days']
            service_level_factor = round(erfinv(2 * service_level - 1) * 2**0.5, 2)
            desv_comb = round(((lead_time * desv_per_day * desv_per_day) + (avg_sales_per_day * avg_sales_per_day
                                                                      * desv_est_lt_days * desv_est_lt_days)) ** 0.5, 2)

            safety_stock_units = round(service_level_factor * desv_comb, 2)
            reorder_point = round(lead_time * avg_sales_per_day + safety_stock_units, 2)
            safety_stock_days = round(safety_stock_units / avg_sales_per_day, 2) if avg_sales_per_day != 0 else 0

            safety_stock = {
                'Familia': str(product['Family']),
                'Categoria': str(product['Category']),
                'Vendedor': str(product['Salesman']),
                'Subcategoria': str(product['Subcategory']),
                'Cliente': (product['Client']),
                'Región': str(product['Region']),
                'SKU': str(product['SKU']),
                'Descripción': str(product['Description']),
                'Promedio': str(avg_sales_per_day),
                'Desviacion': str(desv_per_day),
                'Coeficiente desviacion': str(round(avg_sales_per_day / desv_per_day, 2)) if desv_per_day != 0 else 0,
                'Tiempo demora': str(lead_time),
                'Variabilidad demora': str(desv_est_lt_days),
                'Nivel servicio': str(service_level),
                'Factor Nivel Servicio': str(service_level_factor),
                'Desviacion combinada': str(desv_comb),
                'Punto reorden': str(reorder_point),
                'Stock Seguridad (días)': str(safety_stock_days),
                'Stock Seguridad (unidad)': str(safety_stock_units)
            }

            final_data.append(safety_stock)

        return final_data

    @staticmethod
    def traffic_light(products):
        count_articles = defaultdict(int)
        sum_sales = defaultdict(float)
        sum_stock = defaultdict(float)

        for product in products:
            avg_sales =  product["Venta diaria histórico"] 
            caracterizacion = product["Caracterización"]
            count_articles[caracterizacion] += 1
            sum_sales[caracterizacion] += float(avg_sales)
            sum_stock[caracterizacion] += float(product["Stock"])

        result = [
            {"Caracterización": key, "Cantidad de productos": count_articles[key],
             "Suma venta diaria": round(sum_sales[key], 2), "Suma de stock": round(sum_stock[key], 2)}
            for key in count_articles
        ]

        total_count_articles = sum(count_articles.values())
        total_sum_sales = sum(sum_sales.values())
        total_sum_stock = sum(sum_stock.values())
        result.append({"Caracterización": "Suma total", "Cantidad de productos": total_count_articles,
                       "Suma venta diaria": round(total_sum_sales, 2), "Suma de stock": round(total_sum_stock, 2)})

        sorted_results = sorted(result, key=lambda item: item["Caracterización"])

        return sorted_results

    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        project_pk = request.data.get('project_id')
        # filters = request.data.get('filters')
        # order = request.data.get('order')
        type_of_stock = request.data.get('type')
        params = request.data.get('params')
        is_forecast = True if params["forecast_or_historical"] == "forecast" else False
        forecast_periods = int(params["forecast_periods"])
        scenario = int(params["scenario_id"]) if params["scenario_id"] is not False or params["scenario_id"] else False

        try:
            safety_stock_is_zero = False
            traffic_light = ""
            tables = self.get_data(project_pk=project_pk, scenario=scenario)

            # if filters == '':
            # else:
            # hsd_table, stock_table = self.get_filtered_data(project_pk=project_pk, filters=filters, is_forecast=is_forecast, scenario=scenario)

            if len(tables["historical"]) != len(tables["stock"]):
                return Response(data={'error': 'stock_hsd_dif_len'}, status=status.HTTP_400_BAD_REQUEST)

            data = self.calculate_avg_desv_varcoefficient(historical=tables["historical"], stock=tables["stock"], forecast=tables["forecast"], 
            forecast_periods=forecast_periods)

            if type_of_stock == 'stock by product':
                final_data, safety_stock = self.calculate_stock(data=data, next_buy_days=int(params["next_buy"]), is_forecast=is_forecast)
                safety_stock_is_zero = safety_stock
                traffic_light = self.traffic_light(products=final_data)

            if type_of_stock == 'safety stock':
                final_data = self.calculate_safety_stock(data=data)

            # if order != "":
                # key = list(order.keys())[0]
                # order_type = order[key]

                # if order_type == 'asc':
                    # final_data = sorted(final_data, key=lambda items: items[key], reverse=False)
                # else:
                    # final_data = sorted(final_data, key=lambda items: items[key], reverse=True) 

            return Response(data={"data": final_data, "is_zero": safety_stock_is_zero, "traffic_light": traffic_light},
            status=status.HTTP_200_OK)

        except ValueError as err:
            if str(err) == 'Historical_not_found':
                return Response(data={'error': 'data_none'}, status=status.HTTP_400_BAD_REQUEST)
            
            if str(err) == 'Stock_data_not_found':
                return Response(data={'error': 'stock_data_none'}, status=status.HTTP_400_BAD_REQUEST)

            else:
                print(str(err))
                traceback.print_exc()
                return Response(data={'error': str(err)}, status=status.HTTP_400_BAD_REQUEST)


class StockByProduct(APIView):
    @staticmethod
    def calculate_stock_by_product(forecast_table: pd.DataFrame, stock_data: pd.DataFrame, max_date: str, sku: str):
        # Filter tables by SKU
        forecast = forecast_table.loc[(forecast_table['sku'] == sku) & (forecast_table['model'] != 'actual')]
        rows = forecast.shape[0]
        if rows == 0:
            raise ValueError("SKU not found")

        if rows != 1:
            raise ValueError("More than one product with the same SKU")

        stock = stock_data.loc[stock_data['SKU'] == sku]

        # Get indexes
        max_date_index = forecast_table.columns.get_loc(key=max_date)
        val_last_date_historical = forecast[max_date].values

        # Merge dataframes
        forecast = forecast.iloc[:, max_date_index + 1:]
        dates = forecast.columns.tolist()
        stock = stock.iloc[:, 8:]

        # Get stock data
        available_stock = float(stock['Available Stock'].values)
        safety_stock = float(stock['safety stock'].values)
        purchase_order = float(stock['Purchase order'].values)

        # Calculate
        sales = [round(value, 2) for value in forecast.values.tolist()[0]]
        stock = [available_stock]

        for sale in sales:
            stock_val = round(available_stock - sale - safety_stock + purchase_order, 2)
            stock.append(stock_val)
            available_stock = stock_val
            purchase_order = 0
            safety_stock = 0

        dates.insert(0, max_date)
        sales.insert(0, val_last_date_historical[0])
        return {'sales': sales[:-1], 'stock': stock[:-1], 'dates': dates[:-1]}

    @staticmethod
    def obtain_stock_data(project: int) -> pd.DataFrame | None:
        stock_data = FileRefModel.objects.filter(project_id=project, model_type_id=4).first()

        if stock_data is not None:
            table = pd.read_sql_table(table_name=stock_data.file_name, con=engine)
            return table
        else:
            return None

    @staticmethod
    def obtain_forecast_data_table(scenario: int) -> pd.DataFrame | None:
        forecast_data = ForecastScenario.objects.get(pk=scenario)

        if forecast_data is not None:
            table = pd.read_sql_table(table_name=forecast_data.predictions_table_name, con=engine)
            return table
        else:
            return None

    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def get(self, request):
        project_id = request.query_params.get('project_id')

        if project_id:
            stock_data = self.obtain_stock_data(project=project_id)
            if stock_data is not None:
                return Response(data={'message': 'stock_data_uploaded'}, status=status.HTTP_200_OK)

            else:
                return Response(data={'error': 'stock_data_not_found'}, status=status.HTTP_400_BAD_REQUEST)

        else:
            return Response(data={'error': 'project_id_not_provided'}, status=status.HTTP_400_BAD_REQUEST)

    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        scenario_id = request.data.get('scenario_id')
        project_id = request.data.get('project_id')
        sku = request.data.get('sku')

        forecast = self.obtain_forecast_data_table(scenario=scenario_id)
        stock = self.obtain_stock_data(project=project_id)

        if forecast is not None:

            if stock is not None:
                scenario = ForecastScenario.objects.get(pk=scenario_id)
                max_historical_date = scenario.max_historical_date.strftime('%Y-%m-%d')

                try:
                    stock_by_product = self.calculate_stock_by_product(forecast_table=forecast,
                                                                       stock_data=stock,
                                                                       max_date=max_historical_date,
                                                                       sku=sku)

                    return Response(data={'data': stock_by_product}, status=status.HTTP_200_OK)

                except ValueError as err:
                    if str(err) == "SKU not found":
                        return Response(data={'error': 'sku_not_found'}, status=status.HTTP_400_BAD_REQUEST)

                    elif str(err) == "More than one product with the same SKU":
                        return Response(data={'error': 'multiple_products_with_the_same_sku'},
                                        status=status.HTTP_400_BAD_REQUEST)

            else:
                return Response(data={'error': 'stock_data_not_found'}, status=status.HTTP_400_BAD_REQUEST)

        else:
            return Response(data={'error': 'historical_data_not_found'}, status=status.HTTP_400_BAD_REQUEST)







"""
@staticmethod
def get_filtered_data(project_pk: int, filters: list, is_forecast: bool, scenario: int = None):
    conditions = [
        f"{filter_name} = '{filter_value}'"
        for filter_dict in filters
        for filter_name, filter_value in filter_dict.items()
    ]

    if is_forecast:
        data = ForecastScenario.objects.get(pk=scenario)
        query_for_data = f"SELECT * FROM {data.predictions_table_name} WHERE " + " AND ".join(conditions)
        
    else:
        data = FileRefModel.objects.filter(project_id=project_pk, model_type_id=1).first()
        query_for_data = f"SELECT * FROM {data.file_name} WHERE " + " AND ".join(conditions)

    stock_data = FileRefModel.objects.filter(project_id=project_pk, model_type_id=4).first()

    if data is None:
        raise ValueError("Data_not_found")

    if stock_data is None:
        raise ValueError("Stock_data_not_found")


    query_for_stock = f"SELECT * FROM {stock_data.file_name} WHERE " + " AND ".join(conditions)

    with connection.cursor() as cursor:
        cursor.execute(query_for_data)
        historical_rows = cursor.fetchall()
        columns_historical = [desc[0] for desc in cursor.description]
        df_historical = pd.DataFrame(historical_rows, columns=columns_historical)

        cursor.execute(query_for_stock)
        stock_rows = cursor.fetchall()
        columns_stock = [desc[0] for desc in cursor.description]
        df_stock = pd.DataFrame(stock_rows, columns=columns_stock)

    return df_historical, df_stock 
"""