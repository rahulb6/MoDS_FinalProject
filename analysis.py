import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from statsmodels.tsa.arima.model import ARIMAResults

df = pd.read_csv("food_price_prediction_dataset.csv")
df['Previous_Lag'] = df['Previous_Price'].shift(1).fillna(method='bfill')
df.dropna(inplace=True)
X = df.drop(columns=["Food_Price"])
y = df["Food_Price"]

def evaluate_arima():
    model = joblib.load("arima_model.pkl")
    ts = df.groupby(['Year', 'Month'])['Food_Price'].mean().reset_index()
    ts['date'] = pd.to_datetime(ts[['Year', 'Month']].assign(DAY=1))
    ts.set_index('date', inplace=True)
    actual = ts['Food_Price']
    pred = model.predict(start=0, end=len(actual)-1)
    return compute_metrics(actual, pred, "ARIMA")

def evaluate_lstm():
    ts = df.groupby(['Year', 'Month'])['Food_Price'].mean().reset_index()
    ts['date'] = pd.to_datetime(ts[['Year', 'Month']].assign(DAY=1))
    ts.set_index('date', inplace=True)
    series = ts['Food_Price'].values.reshape(-1, 1)

    scaler = joblib.load("lstm_scaler.pkl")
    series_scaled = scaler.transform(series)

    window = 5
    generator = TimeseriesGenerator(series_scaled, series_scaled, length=window, batch_size=1)
    model = load_model("lstm_model.h5")
    predictions_scaled = model.predict(generator)

    predictions = scaler.inverse_transform(predictions_scaled).flatten()
    actual = series[window:].flatten()

    return compute_metrics(actual, predictions, "LSTM")

def evaluate_xgboost():
    model = joblib.load("xgboost_model.pkl")
    predictions = model.predict(X)
    return compute_metrics(y, predictions, "XGBoost")

def compute_metrics(actual, predicted, name):
    rmse = mean_squared_error(actual, predicted, squared=False)
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    print(f"\n{name} Evaluation")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"MAPE : {mape:.3f}")
    print(f"R^2  : {r2:.3f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=actual, y=predicted)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name}: Actual vs Predicted")
    plt.show()

    return {"model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}

def main():
    results = []
    results.append(evaluate_arima())
    results.append(evaluate_lstm())
    results.append(evaluate_xgboost())

    print("\nComparison Summary:")
    summary = pd.DataFrame(results)
    print(summary.sort_values("RMSE"))

if __name__ == "__main__":
    main()
