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
import math
from scipy.special import erfinv
from datetime import datetime, timedelta
from collections import defaultdict
import traceback
import locale
import math

locale.setlocale(locale.LC_ALL, 'es_ES.utf8')

class StockDataView(APIView):

    # -- METHODS FOR GET DATA FROM DB -- #
    @staticmethod
    def get_data(project_pk: int, only_traffic_light: bool, scenario: int = None, filter_name: str = None, filter_value: str = None):
        try:
            max_historical_date = ""

            if scenario:
                forecast_data = ForecastScenario.objects.get(pk=scenario)
                max_historical_date = forecast_data.max_historical_date.strftime('%Y-%m-%d')
                
                if forecast_data is None:
                    table_forecast = None
            
                else:
                    query = f"SELECT * FROM {forecast_data.predictions_table_name} WHERE model != 'actual'"
                    if only_traffic_light and filter_name and filter_value:
                        query += f" AND {filter_name} = '{filter_value}'"
                    
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

            if only_traffic_light:
                table_historical = table_historical[table_historical[filter_name] == filter_value]
                table_stock = table_stock[table_stock[filter_name] == filter_value]
                # table_forecast = None

            tables = {"historical": table_historical, "stock": table_stock, "forecast": table_forecast}

            return tables, max_historical_date
    
        except Exception as err:
            print("ERROR BUSQUEDA DATOS", err)

    # --- METHODS FOR CALCULATE STOCK --- #
    @staticmethod
    def calculate_avg_desv_varcoefficient(historical: pd.DataFrame, stock: pd.DataFrame, forecast_periods: int, historical_periods: int, forecast, max_hsd):
        try:
            results = []
            iterrows_historical = historical.iloc[:, -historical_periods:].iterrows()
            
            historical.fillna(0, inplace=True)
            stock.fillna(0, inplace=True)

            historical_results = []
            forecast_results = []

            if forecast is not None:
                date_index = forecast.columns.get_loc(max_hsd)
                next_columns = forecast.columns[date_index+1:date_index+1+forecast_periods]
                iterrows_forecast = forecast.loc[:, next_columns].iterrows()
                forecast.fillna(0, inplace=True)

                for idx, row_forecast in iterrows_forecast:
                    total_sales_forecast = row_forecast.sum()
                    avg_row_forecast = np.average(row_forecast)
                    avg_sales_forecast = round(avg_row_forecast / 30, 2)
                    desv_forecast = round(row_forecast.std(), 2)
                    coefficient_of_variation_forecast = round(desv_forecast / avg_sales_forecast, 2) if avg_sales_forecast != 0 else 0
                    stock_or_request_forecast = 'stock' if coefficient_of_variation_forecast > 0.7 else 'request'
                    desv_2_forecast = round(desv_forecast / 30, 2)

                    avg_row_forecast = str(avg_row_forecast) if not pd.isna(avg_row_forecast) else '0'
                    desv_forecast = str(desv_forecast) if not pd.isna(desv_forecast) else '0'
                    coefficient_of_variation_forecast = str(coefficient_of_variation_forecast) if not pd.isna(coefficient_of_variation_forecast) else '0'

                    row_with_stats_forecast = {
                        'index': idx,
                        'total_sales_forecast': total_sales_forecast,
                        'avg_row_forecast': avg_row_forecast,
                        'desv_forecast': desv_forecast,
                        'coefficient_of_variation_forecast': coefficient_of_variation_forecast,
                        'stock_or_request_forecast': stock_or_request_forecast,
                        'avg_sales_per_day_forecast': avg_sales_forecast,
                        'desv_per_day_forecast': desv_2_forecast
                    }

                    forecast_results.append(row_with_stats_forecast)

            for idx, row in iterrows_historical:
                total_sales = row.sum()
                avg_row = np.average(row)
                avg_sales = round(avg_row / 30, 2)
                desv = round(row.std(), 2)
                coefficient_of_variation = round(desv / avg_sales, 2) if avg_sales != 0 else 0
                stock_or_request = 'stock' if coefficient_of_variation > 0.7 else 'request'
                desv_2 = round(desv / 30, 2)

                avg_row = str(avg_row) if not pd.isna(avg_row) else '0'
                desv = str(desv) if not pd.isna(desv) else '0'
                coefficient_of_variation = str(coefficient_of_variation) if not pd.isna(coefficient_of_variation) else '0'

                row_with_stats = {
                    'index': idx,
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

                historical_results.append(row_with_stats)

            # Combina los resultados basados en el índice
            for historical_row in historical_results:
                index = historical_row['index']
                combined_row = historical_row

                # Busca el resultado correspondiente en forecast_results
                forecast_row = next((fr for fr in forecast_results if fr['index'] == index), None)

                if forecast_row:
                    combined_row.update(forecast_row)
                
                # Elimina el índice del resultado final
                del combined_row['index']
                
                results.append(combined_row)
            
            stats_df = pd.DataFrame(results)
        
            result_df = pd.concat(objs=[historical[['SKU', 'Description', 'Family', 'Region', 'Client', 'Salesman', 'Category', 'Subcategory']], stats_df], axis=1)
            merged_df = pd.merge(result_df, stock, on=['SKU', 'Description', 'Family', 'Region', 'Client', 'Salesman', 'Category', 'Subcategory'], how='outer', indicator=True)

            # Convertir la columna '_merge' a categórica
            merged_df['_merge'] = merged_df['_merge'].astype('str')

            # Rellenar los valores 'NaN' con 0 después de convertir la columna '_merge' a categórica
            merged_df.fillna(0, inplace=True)
            result_list = merged_df.to_dict(orient='records')

            return result_list

        except Exception as err:
            print("ERROR CALCULOS", err)
            traceback.print_exc()


    @staticmethod
    def calculate_abc(products, is_forecast):
        try:
            total = sum(product[f'avg_sales_per_day_{"forecast" if is_forecast else "historical"}'] for product in products)
            products.sort(key=lambda x: x[f'avg_sales_per_day_{"forecast" if is_forecast else "historical"}'], reverse=True)
            
            abc_data = []
            acum = 0

            for product in products:
                acum += product[f'avg_sales_per_day_{"forecast" if is_forecast else "historical"}']
                abc_class = 'A' if acum / total <= 0.2 else (
                    'B' if acum / total <= 0.8 else 'C')

                abc = {"SKU": product['SKU'], "ABC": abc_class}
                abc_data.append(abc)

            return abc_data
    
        except Exception as err:
            print("ERROR ABC", err)
    
    @staticmethod
    def calculate_abc_per_category(products: list):
        for product in products:
            product["Price"] = float(product["Price"])

        categories = {}
        abc_data = []
        products.sort(key=lambda product: product["Price"], reverse=True)
    
        for product in products:
            category = product['Category']
            price = product['Price']

            if category in categories:
                categories[category] += price

            else:
                categories[category] = price 
        
        for category, prices in categories.items():
            category_total = np.array(prices)
            percentiles = {
                'A': np.percentile(category_total, 80),
                'B': np.percentile(category_total, 50)
            }
            
        for product in products:
            if product['Category'] == category:
                if product["Price"] >= percentiles['A']:
                    abc_class = "A"
                elif product["Price"] >= percentiles['B']:
                    abc_class = "B"
                else:
                    abc_class = "C"
                
                abc = {"SKU": product["SKU"], "ABC PRECIO": abc_class}
                abc_data.append(abc)

        return abc_data
    
    @staticmethod
    def calculate_optimal_batch(c, d, k):
        c = c if c >= 0 else 0 * 360
        d = int(d)/100
        k = float(f'0.{k}')
        EOQ = math.sqrt((2 * c * d) / k)
        return EOQ

    def calculate_stock(self, data: List[Dict[str, Any]], next_buy_days: int, is_forecast: bool, d, k) -> (
            tuple)[list[dict[str | Any, int | str | datetime | Any]], bool]:
        try:
            def verify_safety_stock_zero(array: List[Dict[str, Any]]):
                for product in array:
                    if product.get("Safety Stock", 0) != 0:
                        return False

                return True
            
            def round_up(n, dec):
                factor = n / dec
                factor = round(factor)
                return factor * dec

            safety_stock_is_zero = verify_safety_stock_zero(data)

            results = []

            abc = self.calculate_abc(products=data, is_forecast=is_forecast)

            abc_dict = {product['SKU']: product['ABC'] for product in abc}
            abc_price_per_category = {product['SKU']: product['ABC PRECIO'] for product in self.calculate_abc_per_category(products=data)}

            for item in data:
                abc_class = abc_dict.get(str(item['SKU']), 'N/A')
                abc_price = abc_price_per_category.get(str(item["SKU"]), "N/a")
                avg_sales_historical = float(item["avg_sales_per_day_historical"])
                price = float(item['Price'])
                avg_sales_forecast = float(item["avg_sales_per_day_forecast"]) 
                purchase_order = float(item['Purchase Order'])
                avg_sales = float(item[f'avg_sales_per_day_{"forecast" if is_forecast else "historical"}'])
                stock = float(item["Stock"])
                available_stock = float(item['Stock']) - float(item['Sales Order Pending Deliverys']) + purchase_order
                lead_time = float(item['Lead Time'])
                safety_stock = float(item['Safety stock (days)'])
                reorder_point = next_buy_days + lead_time + safety_stock
                days_of_coverage = round(available_stock / avg_sales) if avg_sales != 0 else 9999
                buy = 'Si' if (days_of_coverage - reorder_point) < 1 else 'No'
                optimal_batch = float(item["EOQ (Economical order quantity)"])
                how_much = max(optimal_batch, (next_buy_days + lead_time + safety_stock - days_of_coverage) * avg_sales ) if buy == 'Si' else 0
                overflow_units = stock if avg_sales == 0 else (0 if days_of_coverage - reorder_point < 0 else round((days_of_coverage - reorder_point)*avg_sales)) 
                overflow_price = round(overflow_units*price)
                lot_sizing = float(item['Lot Sizing'])
                sales_order = float(item['Sales Order Pending Deliverys'])
                is_obs = str(item['Slow moving'])
                purchase_unit = float(item['Purchase unit'])
                make_to_order = str(item['Make to order'])
                merge = str(item['_merge'])
                
                try:
                    next_buy = datetime.now() + timedelta(days=days_of_coverage - lead_time) if days_of_coverage != 0 \
                        else datetime.now()

                except OverflowError:
                    next_buy = ""
                

                if days_of_coverage == 9999:
                    stock_status = "Obsoleto"
                    if available_stock != 0:
                        characterization = "0-Con stock sin ventas"
                    else:
                        characterization = "Sin stock"
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
             
                print(f"SKU {item['SKU']}: Caracterizacion {characterization}")
    
                next_buy = next_buy.strftime('%Y-%m-%d') if isinstance(next_buy, datetime) else next_buy
                how_much_vs_lot_sizing = round_up(how_much, int(lot_sizing)) if int(lot_sizing) != 0.0 else how_much
                how_much_vs_lot_sizing = max(how_much_vs_lot_sizing, optimal_batch)
                final_how_much = available_stock - sales_order + purchase_order if make_to_order == 'MTO' else round(how_much_vs_lot_sizing) if buy == 'Si' else 0
                final_buy = ('Si' if available_stock - sales_order + purchase_order < 0 else 'No') if make_to_order == 'MTO' else buy

                optimal_batch_calc = self.calculate_optimal_batch(c=avg_sales, d=d, k=k)
                
                try:
                    thirty_days = days_of_coverage - 30 + round(how_much) / avg_sales
                    
                    if thirty_days < reorder_point:
                        calc = optimal_batch_calc / avg_sales
                        if calc < 30:
                            thirty_days = round(30 / calc) * optimal_batch_calc
                        else:
                            thirty_days = optimal_batch_calc 

                except:
                    thirty_days = 0
                
                try:
                    sixty_days = days_of_coverage - 60 + round(how_much) / avg_sales + thirty_days / avg_sales 
                    
                    if sixty_days < reorder_point:
                        calc = optimal_batch_calc / avg_sales
                        if calc < 30:
                            sixty_days = round(30 / calc) * optimal_batch_calc
                        else:
                            sixty_days = optimal_batch_calc 
                
                except:
                    sixty_days = 0

                try:
                    ninety_days = days_of_coverage - 90 + round(how_much) / avg_sales + thirty_days / avg_sales + sixty_days / avg_sales

                    if ninety_days < reorder_point:
                        calc = optimal_batch_calc / avg_sales
                        if calc < 30:
                            ninety_days = round(30 / calc) * optimal_batch_calc
                        else:    
                            ninety_days = optimal_batch_calc
                
                except:
                    ninety_days = 0

                stock = {
                    'Familia': item['Family'],
                    'Categoria': item['Category'],
                    'Vendedor': item['Salesman'],
                    'Subcategoria': item['Subcategory'],
                    'Cliente': item['Client'],
                    'Región': item['Region'],
                    'SKU': str(item['SKU']),
                    'Descripción': str(item['Description']),
                    'Stock': locale.format_string("%d", int(round(stock)), grouping=True),
                    'Stock disponible': locale.format_string("%d", int(round(available_stock)), grouping=True),
                    'EOQ (Calculado)': locale.format_string("%d",optimal_batch_calc, grouping=True),
                    'Ordenes de venta pendientes': sales_order,
                    'Ordenes de compra': purchase_order,
                    'Venta diaria histórico': avg_sales_historical,
                    'Venta diaria predecido': avg_sales_forecast,
                    'Cobertura (días)': str(days_of_coverage),
                    'Punto de reorden': str(reorder_point),
                    '¿Compro?': str(final_buy) if is_obs != 'OB' else 'No',
                    '¿Cuanto?': locale.format_string("%d", round(how_much), grouping=True) if buy == 'Si' and is_obs != 'OB' else "0" ,
                    '¿Cuanto? (Lot Sizing)': locale.format_string("%d", round(final_how_much), grouping=True) if buy == 'Si' and is_obs != 'OB' else "0",
                    '¿Cuanto? (Purchase Unit)': locale.format_string("%d", round(final_how_much * purchase_unit), grouping=True) if buy == 'Si' and is_obs != 'OB' else "0",
                    'Compra 30 días':  0 if make_to_order == "MTO" or is_obs == "OB" else locale.format_string("%d",thirty_days, grouping=True),
                    'Compra 60 días' : 0 if make_to_order == "MTO" or is_obs == "OB" else locale.format_string("%d",sixty_days, grouping=True),
                    'Compra 90 días': 0 if make_to_order == "MTO" or is_obs == "OB" else locale.format_string("%d",ninety_days, grouping=True),
                    'Estado': str(stock_status),
                    'Venta valorizada': locale.format_string("%d", int(round(price * avg_sales)), grouping=True),
                    'Valorizado': locale.format_string("%d", int(round(price * stock)), grouping=True),
                    'Demora en dias': str(lead_time),
                    'Fecha próx. compra': str(next_buy) if days_of_coverage != 9999 else "---",
                    'Caracterización': characterization if merge == 'both' else ('No encontrado en Stock Data' if merge == 'left_only' else 'No encontrado en Historical Data'),
                    'Sobrante (unidades)': locale.format_string("%d", overflow_units, grouping=True),
                    'Cobertura prox. compra (días)': str(days_of_coverage - next_buy_days),
                    'Sobrante valorizado': locale.format_string("%d", int(round(overflow_price)), grouping=True),
                    'Lote optimo de compra': optimal_batch,
                    'Stock seguridad en dias': str(safety_stock),
                    'Unidad de compra': purchase_unit,
                    'Lote de compra': lot_sizing,
                    'Precio unitario': price,
                    'MTO': make_to_order if make_to_order == 'MTO' else '',
                    'OB': is_obs if is_obs == 'OB' else '',
                    'ABC': abc_class,
                    'ABC por categoria': abc_price,
                    'XYZ': item['XYZ']
                }

                results.append(stock)
            
            # print(results)
        
            return results, safety_stock_is_zero
        except Exception as err:
            print("ERROR CALCULO REAPRO", err)
            traceback.print_exc()

    @staticmethod
    def calculate_safety_stock(data: List[Dict[str, Any]]):
        try:
            final_data = []
            for product in data:
                avg_sales_per_day = float(product['avg_sales_per_day_historical'])
                desv_per_day = float(product['desv_per_day_historical'])
                lead_time = float(product['Lead Time'])
                service_level = float(product['Service Level']) / 100
                desv_est_lt_days = float(product['Desv Est Lt Days'])
                service_level_factor = round(erfinv(2 * service_level - 1) * 2**0.5, 2)
                desv_comb = round(((lead_time * desv_per_day * desv_per_day) + (avg_sales_per_day * avg_sales_per_day* desv_est_lt_days * desv_est_lt_days)) ** 0.5, 2)

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
                    'Coeficiente desviacion': str(round(float(avg_sales_per_day) / float(desv_per_day), 2)) if float(desv_per_day) != 0 else 0,
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
        except Exception as err:
            print("ERROR EN CALCULO STOCK SEGURIDAD", err)

    @staticmethod
    def traffic_light(products):
        try:
            count_articles = defaultdict(int)
            sum_sales = defaultdict(float)
            sum_stock = defaultdict(float)
            sum_valued_sales = defaultdict(int)
            sum_valued_stock = defaultdict(int)
            sum_overflow = defaultdict(int)

            for product in products:
                avg_sales =  product["Venta diaria histórico"]
                caracterizacion = product["Caracterización"]
                count_articles[caracterizacion] += 1
                sum_sales[caracterizacion] += float(avg_sales)
                sum_stock[caracterizacion] += float(product["Stock"])
                sum_valued_sales[caracterizacion] += int(locale.atof(product["Venta valorizada"]))
                sum_valued_stock[caracterizacion] += int(locale.atof(product["Valorizado"]))
                sum_overflow[caracterizacion] += int(locale.atof(product["Sobrante valorizado"]))  

            result = [
                {
                    "Caracterización": key, 
                    "Cantidad de productos": locale.format_string("%d", count_articles[key], grouping=True),
                    "Suma venta diaria": locale.format_string("%d",round(sum_sales[key], 2), grouping=True), 
                    "Suma de stock": locale.format_string("%d",round(sum_stock[key], 2), grouping=True), 
                    "Venta valorizada": locale.format_string("%d",round(sum_valued_sales[key]), grouping=True), 
                    "Stock valorizado": locale.format_string("%d",round(sum_valued_stock[key]), grouping=True),
                    "Sobrante valorizado": locale.format_string("%d",round(sum_overflow[key]), grouping=True),
                }
                for key in count_articles
            ]

            total_count_articles = sum(count_articles.values())
            total_sum_sales = sum(sum_sales.values())
            total_sum_stock = sum(sum_stock.values())
            total_valued_sales = sum(sum_valued_sales.values())
            total_valued_stock = sum(sum_valued_stock.values())
            total_overflow = sum(sum_overflow.values())

            result.append({
                "Caracterización": "Suma total", 
                "Cantidad de productos": locale.format_string("%d", total_count_articles, grouping=True),
                "Suma venta diaria": locale.format_string("%d", round(total_sum_sales, 2), grouping=True), 
                "Suma de stock": locale.format_string("%d",round(total_sum_stock, 2), grouping=True), 
                "Venta valorizada": locale.format_string("%d", round(total_valued_sales, 2), grouping=True),
                "Stock valorizado": locale.format_string("%d", round(total_valued_stock, 2), grouping=True),
                "Sobrante valorizado": locale.format_string("%d", round(total_overflow, 2), grouping=True)
            })

            sorted_results = sorted(result, key=lambda item: item["Caracterización"])

            return sorted_results
        except Exception as err:
            print("ERROR EN SEMÁFORO", err)

    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        project_pk = request.data.get('project_id')
        type_of_stock = request.data.get('type')
        params = request.data.get('params')
        historical_periods = int(params["historical_periods"])
        is_forecast = True if params["forecast_or_historical"] == "forecast" else False
        forecast_periods = int(params["forecast_periods"])
        scenario = int(params["scenario_id"]) if params["scenario_id"] is not False or params["scenario_id"] else False
        only_traffic_light = request.GET.get('only_traffic_light', None)
        filter_name = request.GET.get('filter_name', None)
        filter_value = request.GET.get('filter_value', None)
        purchase_cost = params['purchase_cost']
        pruchase_perc = params['purchase_perc']

        try:
            traffic_light = ""
            safety_stock_is_zero = False

            if only_traffic_light == "true":
                tables, max_historical_date = self.get_data(project_pk=project_pk, scenario=scenario, only_traffic_light=True, filter_name=filter_name, filter_value=filter_value)
            
            else:
                tables, max_historical_date = self.get_data(project_pk=project_pk, scenario=scenario, only_traffic_light=False)

            data = self.calculate_avg_desv_varcoefficient(historical=tables["historical"], stock=tables["stock"], forecast=tables["forecast"], 
            forecast_periods=forecast_periods, historical_periods=historical_periods, max_hsd=max_historical_date)

            if type_of_stock == 'stock by product':
                final_data, safety_stock = self.calculate_stock(data=data, next_buy_days=int(params["next_buy"]), is_forecast=is_forecast, d=purchase_cost, k=pruchase_perc)
                safety_stock_is_zero = safety_stock
                traffic_light = self.traffic_light(products=final_data)

            if type_of_stock == 'safety stock':
                final_data = self.calculate_safety_stock(data=data)

            if only_traffic_light == "true":
                return Response(data={"traffic_light": traffic_light},
                status=status.HTTP_200_OK)
            
            else:
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