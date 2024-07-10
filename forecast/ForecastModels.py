import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.linear_model import Lasso, LinearRegression, BayesianRidge
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from prophet import Prophet
import numpy as np


class ForecastModels:

    @staticmethod
    def holt_holtwinters_ema(idx, row, test_periods, prediction_periods, model_name, seasonal_periods):
        time_series = pd.Series(row).astype(dtype='float')

        model = ''

        if model_name == 'holtsWintersExponentialSmoothing':
            try:
                seasonal_periods = int(seasonal_periods)
            except:
                seasonal_periods = float(seasonal_periods)
                seasonal_periods = int(seasonal_periods)
            
            model = ExponentialSmoothing(time_series, seasonal_periods=int(seasonal_periods), trend='add', seasonal='add')

        if model_name == 'holtsExponentialSmoothing':
            model = ExponentialSmoothing(time_series, trend='add')

        if model_name == 'exponential_moving_average':
            model = ExponentialSmoothing(time_series, trend=None, seasonal=None)

        model_fit = model.fit()

        train_pred = model_fit.predict(0)
        forecast = model_fit.forecast(prediction_periods)

        return idx, forecast.tolist() + train_pred.tolist()

    @staticmethod
    def exponential_smoothing(idx, row, test_periods, prediction_periods):
        time_series = pd.Series(row).astype(dtype='float')
        train_data = time_series[:-test_periods]
        test_data = time_series.iloc[-test_periods:]
        model = SimpleExpSmoothing(train_data)

        model_fit = model.fit()

        test_predictions = model_fit.forecast(test_periods)
        train_predictions = model_fit.fittedvalues
        smoothed_series = time_series.ewm(span=10, min_periods=0).mean()
        future_predictions = smoothed_series.iloc[-1:].repeat(prediction_periods)

        return idx, list(train_predictions.values) + list(test_predictions.values) + list(future_predictions.values)

    @staticmethod
    def arima(idx, row, test_periods, prediction_periods, additional_params):

        if additional_params is not None:
            p, d, q = additional_params[f'arima_params']
            arima_order = (int(p), int(d), int(q))

        else:
            arima_order = (0, 0, 0)

        time_series = pd.Series(row).astype(dtype='float')
        train_data = time_series[:-test_periods]
        test_data = time_series.iloc[-test_periods:]
        n_train = len(train_data)

        model = ARIMA(train_data, order=arima_order)
        model.initialize_approximate_diffuse()
        model_fit = model.fit()
        train_predictions = model_fit.predict(start=0, end=n_train - 1)
        test_predictions = model_fit.predict(start=n_train, end=len(time_series) - 1)
        future_predictions = model_fit.forecast(steps=prediction_periods)

        results = list(train_predictions.values) + list(test_predictions.values) + list(future_predictions.values)

        return idx, results

    @staticmethod
    def sarima(idx, row, test_periods, prediction_periods, additional_params):

        if additional_params is not None:
            p, d, q = additional_params[f'sarima_params']
            sarima_order = (int(p), int(d), int(q))

        else:
            sarima_order = (0, 0, 0)

        time_series = pd.Series(row).astype(dtype='float')
        train_data = time_series[:-test_periods]
        test_data = time_series.iloc[-test_periods:]
        n_train = len(train_data)

        model = SARIMAX(train_data, order=sarima_order)
        model.initialize_approximate_diffuse()
        model_fit = model.fit()
        train_predictions = model_fit.predict(start=0, end=n_train - 1)
        test_predictions = model_fit.predict(start=n_train, end=len(time_series) - 1)
        future_predictions = model_fit.forecast(steps=prediction_periods)

        results = list(train_predictions.values) + list(test_predictions.values) + list(future_predictions.values)

        return idx, results

    @staticmethod
    def prophet(idx, row, prediction_periods, additional_params, seasonal_periods, dates, detect_outliers):

        try:
            seasonal_periods = int(seasonal_periods)
        except:
            seasonal_periods = float(seasonal_periods)
            seasonal_periods = int(seasonal_periods)

        def detect_outliers_func(series, threshold=3):
            mean = series.mean()
            std_dev = series.std()
            z_scores = (series - mean) / std_dev
            return z_scores.abs() > threshold
    
        df = pd.DataFrame({'ds': pd.to_datetime(dates), 'y': row})
        df['floor'] = 0
        avg_historical = df['y'].mean()
        max_cap = avg_historical * 2

        df['cap'] = max_cap

        if additional_params is not None:
            seasonality_mode = additional_params[0]
            seasonality_prior_scale = float(additional_params[1])
            uncertainty_samples = int(additional_params[2])
            changepoint_prior_scale = float(additional_params[3])

        else:
            seasonality_mode = "additive"
            seasonality_prior_scale = 10.0
            uncertainty_samples = 1000
            changepoint_prior_scale = 0.05
        
        if detect_outliers:
            outliers = detect_outliers_func(df['y'])
            df['outliers'] = outliers

        model = Prophet(weekly_seasonality=False,
                        yearly_seasonality=seasonal_periods,
                        seasonality_mode=seasonality_mode,
                        seasonality_prior_scale=seasonality_prior_scale,
                        changepoint_prior_scale=changepoint_prior_scale,
                        uncertainty_samples=uncertainty_samples
                        )

        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        if detect_outliers:
            # Exclude outliers during fitting
            model.fit(df[~df['outliers']])

        future = model.make_future_dataframe(periods=prediction_periods, freq='MS')
        future['floor'] = 0
        future['cap'] = max_cap

        forecast = model.predict(future)

        train_predictions_df = model.predict(df)
        train_predictions = train_predictions_df[['ds', 'yhat']].tail(len(dates)).values

        train_predictions_df = model.predict(df)
        train_predictions = train_predictions_df['yhat'].tail(len(dates)).values

        future_predictions = forecast['yhat_lower'].tail(prediction_periods).values

        future_predictions = [max(pred, 0) for pred in future_predictions]
        future_predictions = [min(pred, max_cap) for pred in future_predictions]

        return idx, list(train_predictions) + list(future_predictions)

    @staticmethod
    def lasso(idx, row, test_periods, prediction_periods):
        time_series = pd.Series(row).astype(dtype='float')
        train_data = time_series[:-test_periods]
        test_data = time_series.iloc[-test_periods:]

        model = Lasso(alpha=1.0)
        x_train = pd.DataFrame(pd.to_numeric(pd.to_datetime(train_data.index))).astype(int).values.reshape(-1, 1)
        y_train = train_data.values
        model.fit(x_train, y_train)

        x_test = pd.DataFrame(pd.to_numeric(pd.to_datetime(test_data.index))).astype(int).values.reshape(-1, 1)
        test_predictions = model.predict(x_test)
        train_predictions = model.predict(x_train)

        last_date = pd.to_datetime(time_series.index[-1])
        future_dates = [last_date + pd.DateOffset(days=i) for i in range(1, prediction_periods + 1)]
        x_future = pd.DataFrame(pd.to_numeric(pd.to_datetime(future_dates))).astype(int).values.reshape(-1, 1)
        future_predictions = model.predict(x_future)

        return idx, list(train_predictions) + list(test_predictions) + list(future_predictions)

    @staticmethod
    def decision_tree(idx, row, test_periods, prediction_periods, dates):
        try:
            # Crear una serie de tiempo con índices de fecha
            dates = pd.date_range(start=dates[0], periods=len(row), freq='MS')
            time_series = pd.Series(row, index=dates).astype(dtype='float')
            
            # Dividir en datos de entrenamiento y prueba
            train_data = time_series[:-test_periods]
            test_data = time_series.iloc[-test_periods:]

            # Crear características adicionales
            def create_features(index):
                features = pd.DataFrame(index=index)
                features['month'] = index.month
                features['timestamp'] = pd.to_numeric(index)
                return features

            # Normalizar los datos
            scaler_y = MinMaxScaler()
            y_train = train_data.values.reshape(-1, 1)
            y_train_scaled = scaler_y.fit_transform(y_train)

            x_train = create_features(train_data.index)
            x_test = create_features(test_data.index)

            scaler_x = MinMaxScaler()
            x_train_scaled = scaler_x.fit_transform(x_train)
            x_test_scaled = scaler_x.transform(x_test)

            model = DecisionTreeRegressor(random_state=42, max_depth=5)
            model.fit(x_train_scaled, y_train_scaled)

            test_predictions_scaled = model.predict(x_test_scaled)
            test_predictions = np.squeeze(scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)))
            train_predictions_scaled = model.predict(x_train_scaled)
            train_predictions = np.squeeze(scaler_y.inverse_transform(train_predictions_scaled.reshape(-1, 1)))

            last_date = pd.to_datetime(time_series.index[-1])
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=prediction_periods, freq='MS')
            x_future = create_features(future_dates)
            x_future_scaled = scaler_x.transform(x_future)
            
            future_predictions_scaled = model.predict(x_future_scaled)
            future_predictions = np.squeeze(scaler_y.inverse_transform(future_predictions_scaled.reshape(-1, 1)))
            
            # Evitar predicciones negativas
            future_predictions = [max(pred, 0) for pred in future_predictions]

            return idx, list(train_predictions) + list(test_predictions) + list(future_predictions)

        except Exception as err:
            return str(err)
       

    @staticmethod
    def bayesian(idx, row, test_periods, prediction_periods, dates):
        try:
            # Crear una serie de tiempo con índices de fecha
            dates = pd.date_range(start=dates[0], periods=len(row), freq='MS')
            time_series = pd.Series(row, index=dates).astype(dtype='float')
            
            # Dividir en datos de entrenamiento y prueba
            train_data = time_series[:-test_periods]
            test_data = time_series.iloc[-test_periods:]

            # Normalizar las fechas y los datos
            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()

            x_train = pd.to_numeric(pd.to_datetime(train_data.index)).values.reshape(-1, 1)
            y_train = train_data.values.reshape(-1, 1)
            
            x_train_scaled = scaler_x.fit_transform(x_train)
            y_train_scaled = scaler_y.fit_transform(y_train)

            model = BayesianRidge()
            model.fit(x_train_scaled, y_train_scaled.ravel())

            x_test = pd.to_numeric(pd.to_datetime(test_data.index)).values.reshape(-1, 1)
            x_test_scaled = scaler_x.transform(x_test)

            test_predictions_scaled = model.predict(x_test_scaled)
            test_predictions = np.squeeze(scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)))
            train_predictions_scaled = model.predict(x_train_scaled)
            train_predictions = np.squeeze(scaler_y.inverse_transform(train_predictions_scaled.reshape(-1, 1)))

            last_date = pd.to_datetime(time_series.index[-1])
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, prediction_periods + 1)]
            x_future = pd.to_numeric(pd.to_datetime(future_dates)).values.reshape(-1, 1)
            x_future_scaled = scaler_x.transform(x_future)
            
            future_predictions_scaled = model.predict(x_future_scaled)
            future_predictions = np.squeeze(scaler_y.inverse_transform(future_predictions_scaled.reshape(-1, 1)))
            
            # Evitar predicciones negativas
            future_predictions = [max(pred, 0) for pred in future_predictions]

            return idx, list(train_predictions) + list(test_predictions) + list(future_predictions)

        except Exception as err:
            print(err)

    @staticmethod
    def linear(idx, row, test_periods, prediction_periods, dates):
        try:
            dates = pd.date_range(start=dates[0], periods=len(row), freq='MS')
            time_series = pd.Series(row, index=dates).astype(dtype='float')
            
            # Dividir en datos de entrenamiento y prueba
            train_data = time_series[:-test_periods]
            test_data = time_series.iloc[-test_periods:]

            # Normalizar las fechas y los datos
            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()

            x_train = pd.to_numeric(pd.to_datetime(train_data.index)).values.reshape(-1, 1)
            y_train = train_data.values.reshape(-1, 1)
            
            x_train_scaled = scaler_x.fit_transform(x_train)
            y_train_scaled = scaler_y.fit_transform(y_train)

            model = LinearRegression()
            model.fit(x_train_scaled, y_train_scaled)

            x_test = pd.to_numeric(pd.to_datetime(test_data.index)).values.reshape(-1, 1)
            x_test_scaled = scaler_x.transform(x_test)

            test_predictions_scaled = model.predict(x_test_scaled)
            test_predictions = np.squeeze(scaler_y.inverse_transform(test_predictions_scaled))
            train_predictions_scaled = model.predict(x_train_scaled)
            train_predictions = np.squeeze(scaler_y.inverse_transform(train_predictions_scaled))

            last_date = pd.to_datetime(time_series.index[-1])
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, prediction_periods + 1)]
            x_future = pd.to_numeric(pd.to_datetime(future_dates)).values.reshape(-1, 1)
            x_future_scaled = scaler_x.transform(x_future)
            
            future_predictions_scaled = model.predict(x_future_scaled)
            future_predictions = np.squeeze(scaler_y.inverse_transform(future_predictions_scaled))
            
            # Evitar predicciones negativas
            #future_predictions = [max(pred, 0) for pred in future_predictions]

            return idx, list(train_predictions) + list(test_predictions) + list(future_predictions)
        
        except Exception as err:
            print(err)
