from collections import defaultdict
import pandas as pd
import numpy as np


class Graphic:
    def __init__(self, file_path: str, max_date, pred_p: int, error_method: str):
        self.file_path = file_path
        self.max_date = max_date
        self.pred_p = pred_p
        self.error_method = error_method

    def graphic_predictions(self):
        try:
            df_pred = pd.read_excel(self.file_path)
        except pd.errors.ParserError:
            return {'error': 'file_not_exists'}

        error = df_pred[self.error_method]
        error = np.mean(error)
        error = round(error, 2)
        max_date = pd.to_datetime(self.max_date)
        df_pred = df_pred.drop(columns=[self.error_method])
        date_columns = df_pred.columns[9:]

        actual_rows = df_pred[df_pred['model'] == 'actual']
        other_rows = df_pred[df_pred['model'] != 'actual']

        actual_sum = actual_rows[date_columns].sum()

        other_sum = other_rows[date_columns].sum()

        actual_data = {'x': date_columns.tolist(), 'y': actual_sum.tolist()}

        dates = actual_data['x'][:-self.pred_p]
        values = actual_data['y'][:-self.pred_p]

        actual_data['x'] = dates
        actual_data['y'] = values

        other_data = {'x': date_columns.tolist(), 'y': other_sum.tolist()}

        final_data = {'actual_data': actual_data, 'other_data': other_data}
        data_per_year = self.graphic_predictions_per_year(data=final_data, max_date=max_date)

        return final_data, data_per_year, error

    @staticmethod
    def graphic_predictions_per_year(data: dict, max_date) -> dict:
        actual_data = defaultdict(float)
        other_data = defaultdict(float)
        max_year = pd.to_datetime(max_date).year
        max_month = pd.to_datetime(max_date).month

        for key, value in data.items():
            for date, item_value in zip(value['x'], value['y']):
                year = pd.to_datetime(date).year
                month = pd.to_datetime(date).month

                if key == 'actual_data':
                    actual_data[year] += item_value

                elif key == 'other_data':
                    if year < max_year:
                        other_data[year] = 0

                    elif year == max_year:
                        if month > max_month:
                            other_data[year] += item_value

                        else:
                            other_data[year] = 0

                    else:
                        other_data[year] += item_value

        actual_data = [{'x': year, 'y': value} for year, value in actual_data.items()]
        other_data = [{'x': year, 'y': value} for year, value in other_data.items()]

        actual_data.sort(key=lambda x: x['x'])
        other_data.sort(key=lambda x: x['x'])

        actual_data = [{'x': d['x'], 'y': round(d['y'], 2)} for d in actual_data]
        other_data = [{'x': d['x'], 'y': round(d['y'], 2)} for d in other_data]

        result = {
            "actual_data": {
                "x": [d['x'] for d in actual_data],
                "y": [d['y'] for d in actual_data]
            },
            "other_data": {
                "x": [d['x'] for d in other_data],
                "y": [d['y'] for d in other_data]
            }
        }

        return result