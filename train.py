import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib


def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df, scale_for_nn=False):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df['Previous_Lag'] = df['Previous_Price'].shift(1).fillna(method='bfill')
    df.dropna(inplace=True)

    X = df.drop(columns=["Food_Price"])
    y = df["Food_Price"]

    categorical_features = ["Season"]
    numerical_features = X.drop(columns=categorical_features).columns.tolist()

    scaler = MinMaxScaler() if scale_for_nn else StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical_features),
            ("scale", scaler, numerical_features)
        ]
    )

    return X, y, preprocessor

def train_arima(df):
    ts = df.groupby(['Year', 'Month'])['Food_Price'].mean().reset_index()
    ts['date'] = pd.to_datetime(ts[['Year', 'Month']].assign(DAY=1))
    ts.set_index('date', inplace=True)
    model = ARIMA(ts['Food_Price'], order=(5, 1, 0))
    model_fit = model.fit()
    joblib.dump(model_fit, "arima_model.pkl")
    print("ARIMA model saved as 'arima_model.pkl'")

def train_lstm(df):
    ts = df.groupby(['Year', 'Month'])['Food_Price'].mean().reset_index()
    ts['date'] = pd.to_datetime(ts[['Year', 'Month']].assign(DAY=1))
    ts.set_index('date', inplace=True)
    series = ts['Food_Price'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    window = 5
    generator = TimeseriesGenerator(series_scaled, series_scaled, length=window, batch_size=1)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(generator, epochs=30, verbose=0, callbacks=[EarlyStopping(patience=5)])
    model.save("lstm_model.h5")
    joblib.dump(scaler, "lstm_scaler.pkl")
    print("LSTM model and scaler saved as 'lstm_model.h5' and 'lstm_scaler.pkl'")

def train_xgboost(X, y, preprocessor):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(random_state=42))
    ])

    model.fit(X, y)
    joblib.dump(model, "xgboost_model.pkl")
    print("XGBoost model saved as 'xgboost_model.pkl'")


def main():
    df = load_data("food_price_prediction_dataset.csv")

    train_arima(df)
    train_lstm(df)
    X, y, preprocessor = preprocess_data(df)
    train_xgboost(X, y, preprocessor)

if __name__ == "__main__":
    main()
