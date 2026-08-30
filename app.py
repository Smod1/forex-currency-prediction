import joblib
import keras
import streamlit as st
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet.serialize import model_from_json

st.set_page_config(page_title = "Forex Currency Prediction")
st.title("Forex Currency Prediction By Sarun Modi")
with open("models/model_paths.json", "r") as file:
    model_paths = json.load(file)
currencies = [key for key in model_paths.keys()]

def load_model(currency):
    model_to_load = model_paths[currency]
    path = model_to_load[1]
    name = model_to_load[0]
    # since xgbr, arima, autots were saved using joblib.dump(), joblib.load() will work to load the models
    # but LSTM was saved using keras model.save, Prophet was saved as a .json, so they will each be loaded differently
    match name:
        case "LSTM":
            model = keras.models.load_model(path)
        case "Prophet":
            with open(path, "r") as f:
                model = model_from_json(f.read())
        case "XGBRegressor" | "ARIMA" | "AutoTS":
            model = joblib.load(path)
        case _:
            raise ValueError("Unrecognised model type: ", name)
    return model, name

# create dropdown
chosen_currency = st.selectbox("Choose Currency", currencies)
model, name_of_model = load_model(chosen_currency)

st.text(body = f"Best model for {chosen_currency} is {name_of_model}")

st.header("FORECASTING")
# forecasting horizon input number of days
horizon = st.slider(label = "Choose forecast horizon (days)",
                    min_value = 1,
                    max_value = 60,
                    value = 30, # defaults to forecast 30 days
                    step = 1)

@st.cache_data # eliminates need to reread file from disk every time user moves the slider
def load_data():
    return pd.read_csv("data/cleaned_forex.csv", parse_dates=["date"], index_col="date")

df = load_data()

def forecast(model, model_type, currency, horizon):
    series = df[currency]
    match model_type:
        case "ARIMA":
            values = model.forecast(steps = horizon).values
        case "AutoTS":
            values = model.predict().forecast["value"].values[:horizon]
        case "Prophet":
            future = model.make_future_dataframe(periods = horizon, freq = "D")
            forecast_df = model.predict(future)
            values = forecast_df["yhat"].iloc[-horizon:].to_numpy()
        case "XGBRegressor":
            values = forecast_XGBR(model, series, horizon)
        case "LSTM":
            values = forecast_LSTM(model, series, horizon)
        case _:
            raise ValueError("Unrecognised model type: ", model_type)
    return values

def forecast_XGBR(model, series, horizon):
    # mirrors the recursive forecasting loop inside fit_XGBRegressor
    history = series.copy()
    predictions = []
    for i in range(horizon):
        features = pd.DataFrame(index=[0])
        features["price"] = [history.iloc[-1]]
        features["lag_1"] = [history.iloc[-1]]
        features["lag_2"] = [history.iloc[-2]]
        features["lag_3"] = [history.iloc[-3]]
        features["lag_5"] = [history.iloc[-5]]
        features["lag_10"] = [history.iloc[-10]]

        features["rolling_mean_5"] = [history.iloc[-5:].mean()]
        features["rolling_mean_10"] = [history.iloc[-10:].mean()]
        features["rolling_mean_20"] = [history.iloc[-20:].mean()]

        features["rolling_std_5"] = [history.iloc[-5:].std()]
        features["rolling_std_10"] = [history.iloc[-10:].std()]
        features["rolling_std_20"] = [history.iloc[-20:].std()]

        prediction = model.predict(features)[0]
        predictions.append(prediction)

        # extend history with a placeholder future date so the next iteration's
        # lag/rolling calculations have something to index against
        next_date = history.index[-1] + pd.Timedelta(days=1)
        history = pd.concat([history, pd.Series([prediction], index=[next_date])])

    return np.array(predictions)


def forecast_LSTM(model, series, horizon, lookback=20):
    # mirrors the recursive forecasting loop inside fit_LSTM
    # NOTE: refits a scaler on currently available history
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.to_numpy().reshape(-1, 1))
    history_scaled = scaled[-lookback:].copy()
    predictions_scaled = []
    for i in range(horizon):
        X_input = history_scaled[-lookback:].reshape(1, lookback, 1)
        prediction_scaled = model.predict(X_input, verbose=0)[0, 0]
        predictions_scaled.append(prediction_scaled)
        history_scaled = np.vstack([history_scaled, [[prediction_scaled]]])
    predictions = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    ).flatten()
    return predictions

if st.button("Predict"):
    forecasted_values = forecast(model, name_of_model, chosen_currency, horizon)
    # give chart real future dates on x axis
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_series = pd.Series(forecasted_values, index=future_dates)
    recent_history = df[chosen_currency].iloc[-90:] # gets last 90 real days for context
    combined = pd.concat([recent_history, forecast_series])
    st.text("Forecasted values chart")
    st.line_chart(combined) # displays forecast chart
    st.text("Forecasted values table")
    st.dataframe(combined.rename("Predicted Values"))





