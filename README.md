# forex-currency-prediction
A machine learning and time-series forecasting project for predicting foreign exchange currency prices against the US Dollar (USD).

The project experiments with multiple forecasting approaches — including LSTM, Prophet, XGBoost, ARIMA, and AutoTS — and provides an interactive Streamlit dashboard where users can select a currency and generate forecasts for up to 60 days.

Features
Forex exchange-rate forecasting
Multiple machine learning and time-series models
LSTM neural network forecasting
Prophet time-series forecasting
XGBoost regression
ARIMA forecasting
AutoTS forecasting
Automatic selection of the best-performing model for each currency
Adjustable forecasting horizon from 1–60 days
Interactive forecast visualization
Forecast results displayed in a data table
Docker support
Jupyter Notebook containing the model-development workflow

The overall workflow can be summarized as:

Forex Historical Data
        │
        ▼
Data Cleaning & Preparation
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Train Multiple Forecasting Models
        │
        ├── ARIMA
        ├── AutoTS
        ├── Prophet
        ├── XGBoost
        └── LSTM
        │
        ▼
Model Evaluation
        │
        ▼
Select Best Model per Currency
        │
        ▼
Save Trained Models
        │
        ▼
Streamlit Application
        │
        ▼
Select Currency + Forecast Horizon
        │
        ▼
Generate Forecast
        │
        ▼
Chart + Forecast Table

The application supports five forecasting approaches.

1. ARIMA

ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical time-series model.

The application uses the trained ARIMA model to forecast the requested number of future observations directly.

model.forecast(steps=horizon)
2. AutoTS

AutoTS provides automated time-series modeling and forecasting.

The application loads the saved AutoTS model and retrieves the forecast values for the requested horizon.

3. Prophet

Prophet is a time-series forecasting framework designed to model trends and seasonality.

The application creates a future dataframe and generates predictions:

future = model.make_future_dataframe(
    periods=horizon,
    freq="D"
)

forecast_df = model.predict(future)

The forecasted yhat values are then displayed in the application.

4. XGBoost

The project also uses XGBoost regression with lag-based and rolling statistical features.

Features include:

Current price
Lag 1
Lag 2
Lag 3
Lag 5
Lag 10
5-day rolling mean
10-day rolling mean
20-day rolling mean
5-day rolling standard deviation
10-day rolling standard deviation
20-day rolling standard deviation

For multi-day forecasting, predictions are generated recursively: each new prediction is added to the historical series and subsequently used to construct features for the next prediction.

5. LSTM

The project uses a Long Short-Term Memory (LSTM) neural network for sequential forecasting.

The forecasting process uses a lookback window of 20 observations and generates future predictions recursively.

The LSTM input is structured as:

(batch size, lookback window, features)

with a default lookback of 20 time steps.

The best model was selected using a ranking system over metrics including MAE, RMSE, MAPE

To run this project yourself:
1. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt

Make sure requirements.txt is available in your working copy. If it is not included in the repository, create one containing the dependencies used by the notebook and application.

2. Verify the Required Files

Before running the application, make sure the following files/directories are available:

data/cleaned_forex.csv
models/model_paths.json
models/<trained model files>

The application reads the model configuration from models/model_paths.json when it starts.

3. Run the Application

Start Streamlit:

streamlit run app.py

Streamlit will provide a local URL, typically:

http://localhost:8501

Open the URL in your browser.
