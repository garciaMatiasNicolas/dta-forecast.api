from ..Error import Error
from prophet import Prophet
import pandas as pd
import traceback


def prophet_predictions(row, test_periods, prediction_periods, seasonal_periods, additional_params,
                        error_method, error_periods):
    try:
        df_pred = pd.DataFrame(columns=['Family', 'Region', 'Salesman', 'Client', 'Category', 'Subcategory',
                                        'SKU', 'Description', 'model', 'date', 'value'])

        df_pred_fc = df_pred.copy()
        time_series = pd.Series(row.iloc[12:]).astype(dtype='float')
        train_data = time_series[:-test_periods]
        test_data = time_series.iloc[-test_periods:]

        df_prophet = pd.DataFrame({'ds': train_data.index, 'y': train_data.values})

        (seasonality_mode, seasonality_prior_scale,
         uncertainty_samples, changepoint_prior_scale) = additional_params["prophet_params"]

        model = Prophet(
            yearly_seasonality=int(seasonal_periods),
            seasonality_mode=seasonality_mode,
            seasonality_prior_scale=float(seasonality_prior_scale),
            changepoint_prior_scale=float(changepoint_prior_scale),
            uncertainty_samples=float(uncertainty_samples),
        )
        model.fit(df_prophet)

        # --------------------------------------------------------------------
        start_date = pd.to_datetime(test_data.index[-1])
        next_month = start_date + pd.DateOffset(months=1)
        future_dates = pd.date_range(start=next_month, periods=prediction_periods, freq='MS')

        future_df = pd.DataFrame({'ds': future_dates})

        forecast = model.predict(future_df)

        test_predictions = forecast[-test_periods:]['yhat'].values
        train_predictions = model.predict(df_prophet)['yhat'].values
        # --------------------------------------------------------------------

        copy_fila = row.copy()
        for i, og in enumerate(train_predictions):
            og_date = train_data.index[i]

            df_pred = df_pred._append({
                'Family': row.iloc[0], 'Region': row.iloc[1], 'Salesman': row.iloc[2], 'Client': row.iloc[3],
                'Category': row.iloc[4], 'Subcategory': row.iloc[5],
                'SKU': row.iloc[6], 'Description': row.iloc[7], 'model': 'actual',
                'date': og_date, 'value': copy_fila[og_date]
            }, ignore_index=True)

            df_pred = df_pred._append({
                'Family': row.iloc[0], 'Region': row.iloc[1], 'Salesman': row.iloc[2],
                'Client': row.iloc[3],
                'Category': row.iloc[4], 'Subcategory': row.iloc[5],
                'SKU': row.iloc[6], 'Description': row.iloc[7], 'model': 'prophet',
                'date': og_date, 'value': (0 if og < 0 else og)
            }, ignore_index=True)

        for i, test in enumerate(test_predictions):
            test_date = test_data.index[i]
            df_pred = df_pred._append({
                'Family': row.iloc[0], 'Region': row.iloc[1], 'Salesman': row.iloc[2],
                'Client': row.iloc[3],
                'Category': row.iloc[4], 'Subcategory': row.iloc[5],
                'SKU': row.iloc[6], 'Description': row.iloc[7], 'model': 'actual',
                'date': test_date, 'value': copy_fila[test_date]
            }, ignore_index=True)

            df_pred = df_pred._append({
                'Family': row.iloc[0], 'Region': row.iloc[1], 'Salesman': row.iloc[2],
                'Client': row.iloc[3],
                'Category': row.iloc[4], 'Subcategory': row.iloc[5],
                'SKU': row.iloc[6], 'Description': row.iloc[7], 'model': 'prophet',
                'date': test_date, 'value': (0.0 if test < 0.0 else test)
            }, ignore_index=True)

        df_pred_pivot = df_pred.pivot_table(values='value', index=['Family', 'Region', 'Salesman', 'Client', 'Category',
                                                                   'Subcategory', 'SKU', 'Description', 'model'],
                                            columns='date')

        error = Error(dataframe=df_pred_pivot, model_name='prophet', error_method=error_method, error_periods=error_periods)
        error_calc, last_error = error.calculate_error()

        future_pred_dates = forecast[-prediction_periods:]['ds'].dt.strftime('%Y-%m-%d')

        for i, fut_date in enumerate(future_pred_dates):
            df_pred_fc = df_pred_fc._append({
                'Family': row.iloc[0], 'Region': row.iloc[1], 'Salesman': row.iloc[2],
                'Client': row.iloc[3],
                'Category': row.iloc[4], 'Subcategory': row.iloc[5],
                'SKU': row.iloc[6], 'Description': row.iloc[7], 'model': 'actual',
                'date': fut_date, 'value': None
            }, ignore_index=True)

            df_pred_fc = df_pred_fc._append({
                'Family': row.iloc[0], 'Region': row.iloc[1], 'Salesman': row.iloc[2],
                'Client': row.iloc[3],
                'Category': row.iloc[4], 'Subcategory': row.iloc[5],
                'SKU': row.iloc[6], 'Description': row.iloc[7], 'model': 'prophet',
                'date': fut_date, 'value': (0 if forecast['yhat'].iloc[-prediction_periods:].values[i] < 0 else forecast['yhat'].iloc[-prediction_periods:].values[i])
            }, ignore_index=True)

        df_pred_fc_pivot = df_pred_fc.pivot_table(values='value',
                                                  index=['Family', 'Region', 'Salesman', 'Client', 'Category',
                                                         'Subcategory', 'SKU', 'Description', 'model'],
                                                  columns='date')

        result = pd.concat([df_pred_pivot, df_pred_fc_pivot], axis=1)

        return result, error_calc, last_error

    except Exception as err:
        traceback.print_exc()
        print(f"Error prophet_pred: {str(err)}")



