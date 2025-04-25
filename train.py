import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib


def load_data(path):
    """
    Reads a CSV file and loads it into a pandas DataFrame.
    """
    data = pd.read_csv(path)
    return data


def preprocess_data(data, scale_for_nn=False):
    """
    Cleans and transforms the dataset, prepares it for modeling.
    """
    data.fillna(data.mean(numeric_only=True), inplace=True)
    data['Previous_Lag'] = data['Previous_Price'].shift(1).bfill()
    data.dropna(inplace=True)

    X = data.drop(columns=["Food_Price"])
    y = data["Food_Price"]

    categorical = ["Season"]
    numerical = X.drop(columns=categorical).columns.tolist()

    scaler = MinMaxScaler() if scale_for_nn else StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical),
            ("scale", scaler, numerical)
        ]
    )

    return X, y, preprocessor


def train_arima(data):
    """
    Trains an ARIMA model on monthly average food prices.
    Saves the trained model to disk using joblib.
    """
    ts = data.groupby(['Year', 'Month'])['Food_Price'].mean().reset_index()
    ts['date'] = pd.to_datetime(ts[['Year', 'Month']].assign(DAY=1))
    ts.set_index('date', inplace=True)

    model = ARIMA(ts['Food_Price'], order=(5, 1, 0))
    model_fit = model.fit()

    joblib.dump(model_fit, "models/arima_model_2.pkl")
    print("✅ ARIMA model '")


def train_lstm(data):
    """
    Trains an LSTM model on scaled time-series food prices.
    Saves both the model and the scaler to disk.
    """
def train_lstm(data):
    """
    Trains an LSTM model on scaled time-series food prices.
    Saves both the model and the scaler to disk.
    """
    ts = data.groupby(['Year', 'Month'])['Food_Price'].mean().reset_index()
    ts['date'] = pd.to_datetime(ts[['Year', 'Month']].assign(DAY=1))
    ts.set_index('date', inplace=True)

    series = ts['Food_Price'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    window = 12
    split_idx = int(len(series_scaled) * 0.9)

    train_data = series_scaled[:split_idx]
    val_data = series_scaled[split_idx - window:]  # include overlap

    train_generator = TimeseriesGenerator(train_data, train_data, length=window, batch_size=1)
    val_generator = TimeseriesGenerator(val_data, val_data, length=window, batch_size=1)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=MeanSquaredError())

    model.fit(train_generator, validation_data=val_generator, epochs=50, verbose=1, callbacks=[EarlyStopping(patience=5)])    
    model.save("models/lstm_model_2.h5")
    joblib.dump(scaler, "models/lstm_scaler_2.pkl")

    print("✅ LSTM model '")


def train_xgboost(X, y, preprocessor):
    """
    Trains an XGBoost regression model on the full dataset.
    Includes preprocessing pipeline and saves it using joblib.
    """
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(random_state=42))
    ])

    model.fit(X, y)
    joblib.dump(model, "models/xgboost_model_2.pkl")
    print("✅ XGBoost model '")


def main():
    """
    Entry point for training and saving all models.
    Executes ARIMA, LSTM, and XGBoost training sequentially.
    """
    df = load_data("food_price_prediction_dataset.csv")

    print("Training ARIMA...")
    train_arima(df)

    print("Training LSTM...")
    train_lstm(df)

    print("Preparing data and training XGBoost...")
    X, y, preprocessor = preprocess_data(df)
    train_xgboost(X, y, preprocessor)


if __name__ == "__main__":
    main()
