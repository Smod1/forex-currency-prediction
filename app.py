import streamlit as st
import pandas as pd
import json
st.set_page_config(page_title = "Forex Currency Prediction")
st.title("Forex Currency Prediction")
with open("models/model_paths.json", "r") as file:
    model_paths = json.load(file)
currencies = [key for key in model_paths.keys()]

# create dropdown
chosen_currency = st.selectbox("Choose Currency", currencies)

