from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import pmdarima as pm
from ..Error import Error
import traceback


def sarimax_predictions(row, test_periods, prediction_periods, seasonal_periods, error_periods,
                        additional_params, error_method, row_exog_data=None, row_exog_projected=None):
    try:

        df_pred = pd.DataFrame(columns=['Family', 'Region', 'Salesman', 'Client', 'Category', 'Subcategory',
                                        'SKU', 'Description', 'model', 'date', 'value'])

        df_pred_fc = df_pred.copy()
        time_series = pd.Series(row.iloc[12:]).astype(dtype='float')
        train_data = time_series[:-test_periods]
        test_data = time_series.iloc[-test_periods:]
        n_train = len(train_data)
        model = pm.auto_arima(train_data)
        arima_order = model.order

        if row_exog_data is None:
            ## SARIMA WITHOUT EXOG DATA ##
            model_name = 'sarima'
            model = SARIMAX(train_data, order=arima_order, seasonal_order=(0, 0, 0, seasonal_periods))
            model.initialize_approximate_diffuse()
            model_fit = model.fit()
            train_predictions = model_fit.predict(start=0, end=n_train - 1)
            test_predictions = model_fit.predict(start=n_train, end=len(time_series) - 1)

            start_date = pd.to_datetime(test_data.index[-1])
            next_month = start_date + pd.DateOffset(months=1)
            future_dates = pd.date_range(start=next_month, periods=prediction_periods, freq='MS')
            future_dates = future_dates.strftime('%Y-%m-%d')
            future_predictions = model_fit.forecast(prediction_periods)

        else:
            model_name = 'sarimax'
            common_dates = set(row_exog_data.keys()).intersection(row.keys())
            common_dates = [date for date in common_dates if
                            pd.to_datetime(date, errors='coerce') is not pd.NaT]

            df_merged_data = pd.DataFrame({
                'Dates': common_dates,
                'Sales': [row.get(date) + row.get(date) for date in common_dates],
                'Variable': [row_exog_data.get(date, [float('nan')]) for date in common_dates]
            })

            start_date = pd.to_datetime(test_data.index[-1])
            next_month = start_date + pd.DateOffset(months=1)
            future_dates = pd.date_range(start=next_month, periods=prediction_periods, freq='MS')
            future_dates = future_dates.strftime('%Y-%m-%d')

            if row_exog_projected is not None:
                id_vars = ['Variable', 'Family', 'Region', 'Salesman', 'Client', 'Category', 'Subcategory', 'SKU']
                value_vars = row_exog_projected.columns.difference(id_vars)

                result = row_exog_projected.melt(id_vars=id_vars, value_vars=value_vars,
                                                 var_name='Date', value_name='Value')

                projected_exog_df = result['Value']
                projected_exog_df.columns = ['Variable']

            else:
                data = {'Dates': future_dates, 'Variable': [0] * len(future_dates)}
                projected_exog_df = pd.DataFrame(data)

            model = SARIMAX(df_merged_data['Sales'], order=arima_order, exog=df_merged_data['Variable'],
                            seasonal_order=(0, 0, 0, seasonal_periods))

            model.initialize_approximate_diffuse()
            model_fit = model.fit(disp=False)
            train_predictions = model_fit.predict(start=0, end=n_train - 1, exog=df_merged_data['Variable'])
            test_predictions = model_fit.predict(start=n_train, end=len(time_series) - 1)

            if row_exog_projected is None:
                future_predictions = model_fit.forecast(steps=prediction_periods, exog=projected_exog_df['Variable'])
            else:
                future_predictions = model_fit.forecast(steps=prediction_periods, exog=projected_exog_df)
        # -------------------------------------------------------------
        train_data_index_as_dates = train_data.index
        train_predictions.index = train_data_index_as_dates

        test_data_index_as_dates = test_data.index
        test_predictions.index = test_data_index_as_dates

        future_predictions.index = future_dates

        for i, og in enumerate(train_predictions):
            og_date = train_data.index[i]

            df_pred = df_pred._append(
                {'Family': row.iloc[0], 'Region': row.iloc[1], 'Salesman': row.iloc[2], 'Client': row.iloc[3],
                 'Category': row.iloc[4], 'Subcategory': row.iloc[5],
                 'SKU': row.iloc[6], 'Description': row.iloc[7], 'model': 'actual',
                 'date': og_date, 'value': row[og_date]}, ignore_index=True)

            df_pred = df_pred._append(
                {'Family': row.iloc[0], 'Region': row.iloc[1], 'Salesman': row.iloc[2],
                 'Client': row.iloc[3],
                 'Category': row.iloc[4], 'Subcategory': row.iloc[5],
                 'SKU': row.iloc[6], 'Description': row.iloc[7], 'model': model_name,
                 'date': og_date, 'value': (0 if og < 0 else og)}, ignore_index=True)

        for i, test in enumerate(test_predictions):
            test_date = test_data.index[i]
            df_pred = df_pred._append(
                {'Family': row.iloc[0], 'Region': row.iloc[1], 'Salesman': row.iloc[2],
                 'Client': row.iloc[3], 'Category': row.iloc[4], 'Subcategory': row.iloc[5],
                 'SKU': row.iloc[6], 'Description': row.iloc[7], 'model': 'actual',
                 'date': test_date, 'value': row[test_date]}, ignore_index=True)

            df_pred = df_pred._append(
                {'Family': row.iloc[0], 'Region': row.iloc[1], 'Salesman': row.iloc[2],
                 'Client': row.iloc[3], 'Category': row.iloc[4], 'Subcategory': row.iloc[5],
                 'SKU': row.iloc[6], 'Description': row.iloc[7], 'model': model_name,
                 'date': test_date, 'value': test}, ignore_index=True)

        df_pred_pivot = df_pred.pivot(values='value', index=['Family', 'Region', 'Salesman', 'Client', 'Category',
                                                             'Subcategory', 'SKU', 'Description', 'model'],
                                      columns='date')

        error = Error(dataframe=df_pred_pivot, model_name=model_name, error_method=error_method, error_periods=error_periods)
        error_calc, last_error = error.calculate_error()

        for i, future in enumerate(future_dates):
            fut_date = future_dates[i]
            df_pred_fc = df_pred_fc._append(
                {'Family': row.iloc[0], 'Region': row.iloc[1], 'Salesman': row.iloc[2],
                 'Client': row.iloc[3],
                 'Category': row.iloc[4], 'Subcategory': row.iloc[5],
                 'SKU': row.iloc[6], 'Description': row.iloc[7], 'model': 'actual',
                 'date': fut_date, 'value': None}, ignore_index=True)

            df_pred_fc = df_pred_fc._append(
                {'Family': row.iloc[0], 'Region': row.iloc[1], 'Salesman': row.iloc[2],
                 'Client': row.iloc[3],
                 'Category': row.iloc[4], 'Subcategory': row.iloc[5],
                 'SKU': row.iloc[6], 'Description': row.iloc[7], 'model': model_name,
                 'date': fut_date, 'value': (0 if future_predictions[i] < 0 else future_predictions[i])},
                ignore_index=True)

        df_pred_fc_pivot = df_pred_fc.pivot(values='value', index=['Family', 'Region', 'Salesman', 'Client', 'Category',
                                                                   'Subcategory', 'SKU', 'Description', 'model'],
                                            columns='date')

        result = pd.concat([df_pred_pivot, df_pred_fc_pivot], axis=1)

        return result, error_calc, last_error

    except Exception as err:
        traceback.print_exc()
        print(f"Error arima : {str(err)}")
