from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from ..serializer import FilterData
from ..models import ForecastScenario
from django.db import connection
from datetime import datetime


class ReportDataViews(APIView):

    @staticmethod
    def calc_perc(n1: float, n2: float) -> float:
        try:
            result = round((n1 / n2 - 1) * 100)
            return result
        except ZeroDivisionError:
            return 0

    @staticmethod
    def join_dates(list_dates: list, for_report: bool):
        if for_report:
            dates_joined = " + ".join([f"SUM(\"{date}\")" for date in list_dates])
        else:
            dates_joined = ",\n".join([f"SUM(\"{date}\") as \"{date.split('-')[0]}\"" for date in list_dates])

        return dates_joined

    @staticmethod
    def filter_dates_by_month(last_date, date_list, target_month):
        filtered_dates = []  # List for dates matching the criteria (month and before the current date)
        future_dates = []  # List for dates after the current date

        # Convert last_date to a datetime object (assuming last_date is a date object)
        last_date = datetime.combine(last_date, datetime.min.time())

        for date_str in date_list:
            # Try to parse the date string in two formats (with and without time)
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                date = datetime.strptime(date_str, "%Y-%m-%d")

            if date.month == target_month:  # Check if the date's month matches the target month
                if date < last_date:  # If the date is before the last_date (converted to datetime)
                    filtered_dates.append(date_str)  # Add it to the filtered dates list
                else:
                    future_dates.append(date_str)  # If the date is in the future, add it to the future dates list

        return filtered_dates, future_dates

    def handle_reports(self, filter_name, predictions_table_name, last_date_index, list_date_columns, product=None):
        with (connection.cursor() as cursor):
            last_year_since_last_date = list_date_columns[last_date_index - 12:last_date_index + 1][1:]
            last_quarter_since_last_date = list_date_columns[last_date_index - 3:last_date_index + 1]
            last_month = list_date_columns[last_date_index]

            next_year_since_last_date = list_date_columns[last_date_index + 1:last_date_index + 13]
            next_quarter_since_last_date = list_date_columns[last_date_index + 1:last_date_index + 5]
            next_month_since_last_date = list_date_columns[last_date_index + 1:last_date_index + 2]

            dates_a = list_date_columns[last_date_index - 23:last_date_index - 11]
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
                dates_report = self.join_dates(list_dates=date_range, for_report=True)
                reports_data[date_name] = dates_report

            actual_dates = f'''
                SELECT
                    {'SKU || " " ||DESCRIPTION' if filter_name == "SKU" else filter_name},
                    ROUND({reports_data["last_year_since_last_date"]}),
                    ROUND({reports_data["last_quarter_since_last_date"]}),
                    ROUND(SUM("{last_month}")),
                    ROUND({reports_data["dates_a"]}),
                    ROUND({reports_data["dates_b"]}),
                    ROUND(SUM("{dates_c}")),
                    ROUND({reports_data["dates_d"]}),
                    ROUND(SUM("{dates_e}"))
                FROM {predictions_table_name}
                WHERE model = 'actual'
                GROUP BY {filter_name} UNION 
                SELECT 'TOTAL',
                    ROUND({reports_data["last_year_since_last_date"]}),
                    ROUND({reports_data["last_quarter_since_last_date"]}),
                    ROUND(SUM("{last_month}")),
                    ROUND({reports_data["dates_a"]}),
                    ROUND({reports_data["dates_b"]}),
                    ROUND(SUM("{dates_c}")),
                    ROUND({reports_data["dates_d"]}),
                    ROUND(SUM("{dates_e}"))
                FROM {predictions_table_name}
                WHERE model = 'actual'
                ;
            '''

            predicted_dates = f'''
                SELECT
                    {'SKU || " " ||DESCRIPTION' if filter_name == "SKU" else filter_name},
                    ROUND({reports_data["next_year_since_last_date"]}),
                    ROUND({reports_data["next_quarter_since_last_date"]}),
                    ROUND({reports_data["next_month_since_last_date"]})
                FROM {predictions_table_name}
                WHERE model != 'actual'
                GROUP BY {filter_name} UNION
                SELECT  'TOTAL',
                        ROUND({reports_data["next_year_since_last_date"]}),
                        ROUND({reports_data["next_quarter_since_last_date"]}),
                        ROUND({reports_data["next_month_since_last_date"]})
                FROM {predictions_table_name}
                WHERE model != 'actual';
            '''

            '''
                MYSQL = WITH ROLLUP
            '''

            cursor.execute(sql=actual_dates)
            actual_dates = cursor.fetchall()

            cursor.execute(sql=predicted_dates)
            predicted_dates = cursor.fetchall()

            for i in range(len(actual_dates)):
                category, *actual_values = actual_dates[i]
                for pred_category, *predicted_values in predicted_dates:
                    if category == pred_category:
                        actual_dates[i] = (category, *actual_values, *predicted_values)

            final_data = []

            for item in actual_dates:
                ytd = self.calc_perc(n1=item[1], n2=item[4])
                qtd = self.calc_perc(n1=item[2], n2=item[5])
                mtd = self.calc_perc(n1=item[3], n2=item[6])
                ytg = self.calc_perc(n1=item[9], n2=item[1])
                qtg = self.calc_perc(n1=item[10], n2=item[7])
                mtg = self.calc_perc(n1=item[11], n2=item[8])

                final_data.append([item[0], ytd, qtd, mtd, ytg, qtg, mtg])

            return final_data

    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        filters = FilterData(data=request.data)
        product = request.data.get('product')

        if filters.is_valid():
            scenario_id = filters.validated_data['scenario_id']
            filter_name = filters.validated_data['filter_name']
            month = filters.validated_data['filter_value']
            scenario = ForecastScenario.objects.filter(pk=scenario_id).first()
            predictions_table_name = scenario.predictions_table_name

            with connection.cursor() as cursor:
                last_date = scenario.max_historical_date

                cursor.execute(sql=f'''SELECT name FROM pragma_table_info("{predictions_table_name}") 
                                WHERE name LIKE "%-%";''')
                date_columns = cursor.fetchall()

                """
                SQL QUERY FOR MYSQL
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = NOMBRE_TABLA
                    AND COLUMN_NAME LIKE '%-%';
                """

                # Get the years distinct and get a transform into a list the sqlquery tuple list
                years_set = set()
                list_date_columns = []

                for date in date_columns:
                    date_str = date[0]
                    list_date_columns.append(date_str)
                    year = date_str.split('-')[0]
                    years_set.add(year)

                years = sorted(list(years_set))

                new_last_date = last_date.strftime('%Y-%m-%d')
                last_date_index = list_date_columns.index(new_last_date)

                # Handle reports and get data
                final_data = self.handle_reports(filter_name, predictions_table_name,
                                                 last_date_index, list_date_columns, product)

                past_dates, future_dates = self.filter_dates_by_month(last_date=last_date,
                                                                      date_list=list_date_columns,
                                                                      target_month=int(month))

                past_cols = self.join_dates(list_dates=past_dates, for_report=False)
                future_cols = self.join_dates(list_dates=future_dates, for_report=False)

                query_for_past_dates = f'''
                    SELECT {'SKU || " " ||DESCRIPTION' if filter_name == "SKU" else filter_name},
                        {past_cols}
                    FROM {predictions_table_name}
                    WHERE model = 'actual'
                    {'AND SKU = ' + f"'{str(product)}'" if product else ''}
                    GROUP BY {filter_name};
                '''

                query_for_future_dates = f'''
                    SELECT {'SKU || " " ||DESCRIPTION' if filter_name == "SKU" else filter_name},
                        {future_cols}
                    FROM {predictions_table_name}
                    WHERE model != 'actual'
                    {'AND SKU = ' + f"'{str(product)}'" if product else ''}
                    GROUP BY {filter_name};
                '''

                cursor.execute(sql=query_for_past_dates)
                past_data = cursor.fetchall()

                cursor.execute(sql=query_for_future_dates)
                future_data = cursor.fetchall()

                dict_values = {tupla[0]: tupla[1] for tupla in future_data}

                rounded_data = [[elem[0]] + [round(val) for val in elem[1:]] +
                                [round(dict_values[elem[0]])] for elem in past_data]

                num_sublist = [sublist[1:] for sublist in rounded_data if
                               all(isinstance(item, (int, float)) for item in sublist[1:])]

                total = ['TOTAL'] + [sum(item) for item in zip(*num_sublist)]

                rounded_data.append(total)

                json_data = dict(columns=years, data=rounded_data)

                return Response({"data_per_month": json_data, "reports": final_data},
                                status=status.HTTP_200_OK)

        else:
            print(filters.errors)
            return Response({'error': 'bad_request', 'logs': filters.errors},
                            status=status.HTTP_400_BAD_REQUEST)


class ModelsGraphicAPIView(APIView):
    @authentication_classes([TokenAuthentication])
    @permission_classes([IsAuthenticated])
    def post(self, request):
        scenario_pk = request.data.get('scenario_id')
        scenario = ForecastScenario.objects.get(id=scenario_pk)

        if scenario:
            table_name = scenario.predictions_table_name

            with connection.cursor() as cursor:
                cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
                rows = cursor.fetchall()
                total = rows[0][0] / 2

                cursor.execute(f'''
                    SELECT  
                    MODEL, 
                    COUNT(*) 
                    FROM {table_name}
                    WHERE MODEL != 'actual' GROUP BY MODEL;''')

                data_rows = cursor.fetchall()

                models = []
                avg = []

                for row in data_rows:
                    model, number_model = row[0], row[1]
                    percentage = round((number_model / total) * 100, 2)
                    models.append(model)
                    avg.append(percentage)

            return Response({'models': models, 'avg': avg}, status=status.HTTP_200_OK)

        else:
            return Response({'error': 'scenario_not_found'}, status=status.HTTP_400_BAD_REQUEST)
