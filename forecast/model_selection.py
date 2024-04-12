from .forecast_models import (arima_predictions, linear_regression, exponential_smothing,
    holt_winters_holt_EMA, lasso, bayesian_regression, decision_tree,
    prophet, sarimax_predictions, arimax_predictions)
from database.db_engine import engine
import pandas as pd
import numpy as np
from sqlalchemy.exc import NoSuchTableError


# Function to get historical data
def get_historical_data(table_name: str):
    try:
        dataframe = pd.read_sql_table(table_name, con=engine)

        dataframe.iloc[:, 13:] = dataframe.iloc[:, 13:].replace(to_replace=["NaN", "null", "nan"], value=np.nan)
        dataframe.iloc[:, 13:] = dataframe.iloc[:, 13:].fillna(0).apply(pd.to_numeric, errors='coerce').values

        columns = dataframe.columns[13:]

        dataframe.iloc[:, 13:] = dataframe.iloc[:, 13:].fillna(0)

        return dataframe

    except NoSuchTableError as e:
        return None


# Function to choose best model
def best_model(df_historical: pd.DataFrame, test_p: int, pred_p: int, error_periods: int, models: list, seasonal_periods,
               additional_params: dict, error_method: str, scenario_name: str, exog_dataframe=None,
               exog_projected_df=None):

    df_historical = df_historical.copy()

    df_pred = pd.DataFrame()

    model_data = {}
    last_errors = []

    try:
        seasonal_order_value = int(seasonal_periods)
    except ValueError:
        seasonal_order_value = int(float(seasonal_periods))

    for _, row in df_historical.iterrows():

        if len(row.iloc[12:]) <= test_p:
            raise ValueError("error_test_periods")

        if 'arima' in models:
            arima_df, arima_mape, arima_last_error = arima_predictions.arima_sarima_arimax_sarimax_predictions(row=row,
                                                                                             test_periods=test_p,
                                                                                             prediction_periods=pred_p,
                                                                                             seasonal_periods=seasonal_order_value,
                                                                                             model_name='arima',
                                                                                             additional_params=additional_params,
                                                                                             error_method=error_method,
                                                                                             error_periods=error_periods)

            model_data['arima'] = {error_method: arima_mape, 'df': arima_df}
            last_errors.append(arima_last_error)
        
        if 'holtsWintersExponentialSmoothing' in models:

            try:
                holt_wint_df, holt_wint_mape, holt_wint_last_error = holt_winters_holt_EMA.holts_winters_holts_ema(row=row,
                                                                                             test_periods=test_p,
                                                                                             prediction_periods=pred_p,
                                                                                             additional_params=additional_params,
                                                                                             model_name='holt_winters',
                                                                                             seasonal_periods=seasonal_order_value,
                                                                                             error_method=error_method,
                                                                                             error_periods=error_periods)

                model_data['holtsWintersExponentialSmoothing'] = {error_method: holt_wint_mape, 'df': holt_wint_df}
                last_errors.append(holt_wint_last_error)

            except ValueError as err:
                if str(err) == ('''Cannot compute initial seasonals using heuristic method with 
                less than two full seasonal cycles in the data.'''):
                    return err

        if 'holtsExponentialSmoothing' in models:

            holt_df, holt_mape, holt_last_error = holt_winters_holt_EMA.holts_winters_holts_ema(row=row,
                                                                               test_periods=test_p,
                                                                               prediction_periods=pred_p,
                                                                               additional_params=additional_params,
                                                                               model_name='holt',
                                                                               seasonal_periods=seasonal_order_value,
                                                                               error_method=error_method,
                                                                               error_periods=error_periods)

            model_data['holtsExponentialSmoothing'] = {error_method: holt_mape, 'df': holt_df}
            last_errors.append(holt_last_error)

        if 'exponential_moving_average' in models:
            ema_df, ema_mape, ema_last_error = holt_winters_holt_EMA.holts_winters_holts_ema(row=row, test_periods=test_p,
                                                                             prediction_periods=pred_p,
                                                                             model_name='exponential_moving_average',
                                                                             seasonal_periods=seasonal_order_value,
                                                                             error_method=error_method,
                                                                             error_periods=error_periods)

            model_data['exponential_moving_average'] = {error_method: ema_mape, 'df': ema_df}
            last_errors.append(ema_last_error)
        
        if 'simpleExponentialSmoothing' in models:
            exp_df, exp_mape, exp_last_error = exponential_smothing.exp_smoothing_predictions(row, test_p, pred_p,
                                                                                              seasonal_periods,
                                                                                              error_method=error_method,
                                                                                              error_periods=error_periods)

            model_data['simpleExponentialSmoothing'] = {error_method: exp_mape, 'df': exp_df}
            last_errors.append(exp_last_error)
        
        if 'prophet' in models:
            prophet_df, prophet_mape, prophet_last_error = prophet.prophet_predictions(row=row, test_periods=test_p,
                                                                   prediction_periods=pred_p,
                                                                   seasonal_periods=seasonal_order_value,
                                                                   additional_params=additional_params,
                                                                   error_method=error_method,
                                                                   error_periods=error_periods)

            model_data['prophet'] = {error_method: prophet_mape, 'df': prophet_df}
            last_errors.append(prophet_last_error)
 
        if 'linearRegression' in models:
            linear_df, linear_mape, linear_last_error = linear_regression.linear_regression_predictions(fila=row,
                                                                                     test_periods=test_p,
                                                                                     prediction_periods=pred_p,
                                                                                     seasonal_periods=seasonal_order_value,
                                                                                     error_method=error_method,
                                                                                     error_periods=error_periods)

            model_data['linearRegression'] = {error_method: linear_mape, 'df': linear_df}
            last_errors.append(linear_last_error)

        if 'lasso' in models:
            lasso_df, lasso_mape, lasso_last_error = lasso.lasso_regression_predictions(row, test_p,
                                                                                        pred_p, seasonal_periods,
                                                                                        error_method=error_method,
                                                                                        error_periods=error_periods)

            model_data['lasso'] = {error_method: lasso_mape, 'df': lasso_df}
            last_errors.append(lasso_last_error)
        
        if 'bayesian' in models:
            bayesian_df, bayesian_mape, bayesian_last_error = bayesian_regression.bayesian_regression_predictions(row,
                                                                                             test_p, pred_p,
                                                                                             error_method=error_method,
                                                                                             error_periods=error_periods)

            model_data['bayesian'] = {error_method: bayesian_mape, 'df': bayesian_df}
            last_errors.append(bayesian_last_error)

        if 'decisionTree' in models:
            decision_tree_df, decision_tree_mape, decision_tree_last_error = decision_tree.decision_tree_regression_predictions(
                                                                                                      fila=row,
                                                                                                      test_periods=test_p,
                                                                                                      prediction_periods=pred_p,
                                                                                                      seasonal_periods=seasonal_order_value,
                                                                                                      error_method=error_method,
                                                                                                      error_periods=error_periods)

            model_data['decisionTree'] = {error_method: decision_tree_mape, 'df': decision_tree_df}
            last_errors.append(decision_tree_last_error)
         
        if 'sarima' in models:

            sarima_df, sarima_mape, sarima_last_error = arima_predictions.arima_sarima_arimax_sarimax_predictions(row=row,
                                                                                               test_periods=test_p,
                                                                                               prediction_periods=pred_p,
                                                                                               seasonal_periods=seasonal_order_value,
                                                                                               model_name='sarima',
                                                                                               additional_params=additional_params,
                                                                                               error_method=error_method,
                                                                                               error_periods=error_periods)

            model_data['sarima'] = {error_method: sarima_mape, 'df': sarima_df}
            last_errors.append(sarima_last_error)
            sarima_df = sarima_df.assign(MAPE=lambda x: sarima_mape)

        if exog_dataframe is not None:
            df_exog = exog_dataframe.copy()
            row_exog_data = None

            for _, row_historical in df_historical.iterrows():

                for column_exog, row_exog in df_exog.iterrows():

                    if row_exog['Family'] == 'all_data':
                        row_exog_data = row_exog
                        break

                    elif (
                            (row_exog['Family'] == row_historical['Family']) or
                            (row_exog['Region'] == row_historical['Region']) or
                            (row_exog['Category'] == row_historical['Category']) or
                            (row_exog['Subcategory'] == row_historical['Subcategory']) or
                            (row_exog['Client'] == row_historical['Client']) or
                            (row_exog['Salesman'] == row_historical['Salesman'])
                    ):
                        row_exog_data = row_exog
                        break

                    else:
                        row_exog_data = None

            if 'sarimax' in models:
                sarimax_df, sarimax_mape, sarimax_last_error = sarimax_predictions.sarimax_predictions(
                    row=row, prediction_periods=pred_p, test_periods=test_p,
                    seasonal_periods=seasonal_order_value,
                    row_exog_data=row_exog_data, error_method=error_method,
                    additional_params=additional_params, row_exog_projected=exog_projected_df, error_periods=error_periods)

                model_data['sarimax'] = {error_method: sarimax_mape, 'df': sarimax_df}
                last_errors.append(sarimax_last_error)

            if 'arimax' in models:
                arimax_df, arimax_error, arimax_last_error = arimax_predictions.arimax_predictions(
                    row=row, prediction_periods=pred_p, test_periods=test_p,
                    seasonal_periods=seasonal_order_value, error_method=error_method, error_periods=error_periods)

                model_data['arimax'] = {error_method: arimax_error, 'df': arimax_df}
                last_errors.append(arimax_last_error)

        best_model_name = min(model_data, key=lambda k: model_data[k][error_method])
        best_df = model_data[best_model_name]['df']
        best_df[error_method] = model_data[best_model_name][error_method]
        df_pred = df_pred._append(best_df, ignore_index=False)

    last_errors_avg = round(np.mean(last_errors), 1)

    return df_pred, last_errors_avg
