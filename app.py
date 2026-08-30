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





