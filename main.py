import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer, mean_absolute_percentage_error

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


def load_data(path):
    df = pd.read_csv(path)
    return df

def perform_eda(df):
    print("\nQuick Overview of the Dataset:\n", df.describe())
    df.hist(figsize=(14, 10), bins=20)
    plt.suptitle("Distributions of Numeric Features")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df.select_dtypes(include=np.number), orient="h")
    plt.title("Outliers and Spread across Features")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="Year", y="Food_Price", estimator='mean')
    plt.title("Average Food Price Trends Over Years")
    plt.ylabel("Price (USD)")
    plt.show()

    plt.figure(figsize=(12, 10))
    corr = df.drop(columns=["Season"]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Map of Key Features")
    plt.show()

# Data preparation based on model requirement

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

    return X, y, preprocessor, categorical_features, numerical_features


def train_arima(df):
    import joblib
    ts = df.groupby(['Year', 'Month'])['Food_Price'].mean().reset_index()
    ts['date'] = pd.to_datetime(ts[['Year', 'Month']].assign(DAY=1))
    ts.set_index('date', inplace=True)
    model = ARIMA(ts['Food_Price'], order=(5, 1, 0))
    model_fit = model.fit()
    print(model_fit.summary())
    ts['forecast'] = model_fit.predict(start=0, end=len(ts)-1)
    ts[['Food_Price', 'forecast']].plot(title="ARIMA Forecast vs Actual")
    plt.show()
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

    model.fit(generator, epochs=30, verbose=1, callbacks=[EarlyStopping(patience=5)])
    y_pred = model.predict(generator)

    plt.plot(series_scaled[window:], label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("LSTM - Predicted vs Actual (Scaled)")
    plt.legend()
    plt.show()

    model.save("lstm_model.h5")
    print("LSTM model saved as 'lstm_model.h5'")


def train_xgboost(X, y, preprocessor):
    import joblib
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nXGBoost Evaluation Metrics")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"MAPE : {mape:.3f}")
    print(f"R²   : {r2:.3f}")

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Food Prices")
    plt.ylabel("Predicted Food Prices")
    plt.title("Prediction Performance (XGBoost)")
    plt.show()

    joblib.dump(model, "xgboost_model.pkl")
    print("XGBoost model saved as 'xgboost_model.pkl'")



def main():
    df = load_data('food_price_prediction_dataset.csv')
    perform_eda(df)

    print("\nSelect the model you'd like to run:")
    print("1 - ARIMA")
    print("2 - LSTM")
    print("3 - XGBoost")
    option = input("Enter your choice (1/2/3): ").strip()

    if option == '1':
        train_arima(df)
    elif option == '2':
        train_lstm(df)
    elif option == '3':
        X, y, preprocessor, _, _ = preprocess_data(df)
        train_xgboost(X, y, preprocessor)
    else:
        print("Invalid input. Please choose between 1, 2 or 3.")


if __name__ == "__main__":
    main()
