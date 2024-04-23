import numpy as np
import pandas as pd


class Error:

    def __init__(self, model_name: str, error_method: str, error_periods: int, dataframe: pd.DataFrame = None):
        self.df = dataframe
        self.model_name = model_name
        self.error_method = error_method
        self.error_periods = error_periods

        if self.error_method not in ['MAE', 'MAPE', 'SMAPE', 'RMSE']:
            raise ValueError("Invalid error_method. Use 'MAE', 'MAPE', 'SMAPE', 'RMSE'.")

    def calculate_error(self) -> float:
        absolute_errors = []

        if self.error_periods == 0:
            last_twelve_periods_predicted = self.df.xs(self.model_name, level='model')
            last_twelve_periods_actual = self.df.xs('actual', level='model')

        else:
            last_twelve_periods_predicted = self.df.xs(self.model_name, level='model').iloc[:, -self.error_periods:]
            last_twelve_periods_actual = self.df.xs('actual', level='model').iloc[:, -self.error_periods:]

        for col in last_twelve_periods_predicted.columns:
            predicted_col = last_twelve_periods_predicted[col]
            actual_col = last_twelve_periods_actual[col]
            n = len(actual_col)

            for i in range(n):
                error = 0
                if self.error_method == 'MAPE':
                    error = self.calculate_mape(predicted=predicted_col[i], actual=actual_col[i])

                if self.error_method == 'SMAPE':
                    error = self.calculate_smape(predicted=predicted_col[i], actual=actual_col[i])

                if self.error_method == 'MAE':
                    error = self.calculate_mae(predicted=predicted_col[i], actual=actual_col[i])

                if self.error_method == 'RMSE':
                    error = self.calculate_rmse(predicted=predicted_col[i], actual=actual_col[i])

                absolute_errors.append(error)

        total_error = round(sum(absolute_errors) / len(absolute_errors), 2)
        last_period_error = absolute_errors[-1]

        return total_error, last_period_error

    def calculate_error_last_period(self, prediction_periods: int) -> tuple[float, float]:
        methods = {
            'MAPE': Error.calculate_mape,
            'SMAPE': Error.calculate_smape,
            'RMSE': Error.calculate_rmse,
            'MAE': Error.calculate_mae
        }

        last_period_column = prediction_periods + 2
        last_period = self.df.iloc[:, -last_period_column]

        values = []
        actual_vals = []
        predicted_vals = []

        for i in range(0, len(last_period), 2):
            actual = last_period[i]
            predicted = last_period[i + 1]
            actual_vals.append(actual)
            predicted_vals.append(predicted)
            error = 0

            if self.error_method in methods:
                calc_error = methods[self.error_method]
                error = calc_error(predicted, actual)

            values.append(error)

        absolute_error = 0
        if self.error_method == 'MAPE':
            absolute_error = self.calculate_mape(predicted=sum(predicted_vals), actual=sum(actual_vals))

        if self.error_method == 'SMAPE':
            absolute_error = self.calculate_smape(predicted=sum(predicted_vals), actual=sum(actual_vals))

        if self.error_method == 'MAE':
            absolute_error = self.calculate_mae(predicted=sum(predicted_vals), actual=sum(actual_vals))

        if self.error_method == 'RMSE':
            absolute_error = self.calculate_rmse(predicted=sum(predicted_vals), actual=sum(actual_vals))

        return absolute_error

    @staticmethod
    def calculate_mape(actual: float, predicted: float) -> float:
        if actual == 0 and predicted == 0:
            mape = 0
        elif (actual == 0 or actual < 0) and predicted != 0:
            mape = 100
        else:
            mape = abs((actual - predicted) / actual) * 100

        return round(float(mape), 2)

    @staticmethod
    def calculate_rmse(actual: float, predicted: float) -> float:
        if actual == 0 and predicted == 0:
            rmse = 0
        elif (actual == 0 or actual < 0) and predicted != 0:
            rmse = 100
        else:
            rmse = (actual - predicted) ** 2

        return round(np.sqrt(rmse), 2)

    @staticmethod
    def calculate_mae(actual: float, predicted: float) -> float:
        if actual == 0 and predicted == 0:
            mae = 0
        elif (actual == 0 or actual < 0) and predicted != 0:
            mae = 100
        else:
            mae = abs(actual - predicted)

        return round(mae, 2)

    @staticmethod
    def calculate_smape(actual: float, predicted: float) -> float:
        if actual == 0 and predicted == 0:
            smape = 0
        elif (actual == 0 or actual < 0) and predicted != 0:
            smape = 100
        else:
            smape = abs((actual - predicted) / ((actual + predicted) / 2)) * 100

        return round(smape, 2)



