import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv("food_price_prediction_dataset.csv")
df['Previous_Lag'] = df['Previous_Price'].shift(1).bfill()
df.dropna(inplace=True)

# --------------------------
# 1. EDA Plots
# --------------------------

print("\n✅ Descriptive Statistics:\n")
print(df.describe())

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Distribution of the target variable
plt.figure(figsize=(6, 4))
sns.histplot(df['Food_Price'], kde=True)
plt.title("Distribution of Food Price")
plt.show()

# --------------------------
# 2. Feature Importance: XGBoost
# --------------------------

X = df.drop(columns=['Food_Price'])
y = df['Food_Price']

categorical = ["Season"]
numerical = X.drop(columns=categorical).columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical),
    ("num", StandardScaler(), numerical)
])

X_preprocessed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()

pipeline = joblib.load("models/xgboost_model_3.pkl")
xgb = pipeline.named_steps['regressor']

importances = xgb.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx])
plt.title("Feature Importance Scores (XGBoost)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# --------------------------
# 3. Significance Discussion (print)
# --------------------------

print("\n📌 Top Influential Features toward Food_Price (XGBoost):")
top_features = pd.DataFrame({
    'Feature': np.array(feature_names)[sorted_idx],
    'Importance': importances[sorted_idx]
})
print(top_features.head(10))
