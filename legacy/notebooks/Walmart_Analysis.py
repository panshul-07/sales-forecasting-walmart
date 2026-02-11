# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm

# --- Load Data ---
target_file = 'Walmart Data Analysis and Forcasting.csv'
file_path = None

for root, dirs, files in os.walk(os.getcwd()):
    if target_file in files:
        file_path = os.path.join(root, target_file)
        break

if file_path:
    print(f"✅ Found it at: {file_path}")
    df = pd.read_csv(file_path)

    # --- Feature Engineering ---
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(['Store', 'Date'])

    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Last_Week_Sales'] = df.groupby('Store')['Weekly_Sales'].shift(1)
    df['Rolling_4W_Avg'] = df.groupby('Store')['Weekly_Sales'].transform(
        lambda x: x.rolling(window=4).mean().shift(1)
    )

    df = df.dropna()
    print("✅ Features created successfully")
else:
    raise FileNotFoundError("ERROR: File not found. Please place the CSV in the same folder as this code.")

print(df.info())
print(df.head())

# --- Define Features & Target ---
features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI',
            'Unemployment', 'Month', 'Week', 'Last_Week_Sales', 'Rolling_4W_Avg']

X = df[features]
y = df['Weekly_Sales']

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Models ---
rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
xgb = XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=9, n_jobs=-1)

super_model = VotingRegressor(estimators=[('rf', rf), ('xgb', xgb)])
super_model.fit(X_train, y_train)

# --- Save Model ---
joblib.dump(super_model, 'walmart_demand_model.pkl')
print("✅ Model training complete and saved")

# --- Predictions ---
preds = super_model.predict(X_test)
accuracy = r2_score(y_test, preds)
print(f"🎯 FINAL RESULT: {accuracy*100:.2f}% Accuracy")

# --- Export Results for Tableau ---
tableau_results = X_test.copy()
tableau_results['Actual_Sales'] = y_test
tableau_results['Predicted_Sales'] = preds
tableau_results['Date'] = df.loc[X_test.index, 'Date']
tableau_results.to_csv('Walmart_Final_Results.csv', index=False)
print("✅ Results exported to Walmart_Final_Results.csv")

# --- OLS Regression for Interpretability ---
X_econ = sm.add_constant(X)
econ_model = sm.OLS(y, X_econ).fit()
print(econ_model.summary())
print("\n--- Demand Equation Coefficients ---")
print(econ_model.params)

# --- Visualization ---
plt.figure(figsize=(16, 5))

# Actual vs Predicted
plt.subplot(1, 2, 1)
sns.regplot(x=y_test, y=preds, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Actual vs Predicted Sales')

# Variable Impact (OLS Coefficients)
plt.subplot(1, 2, 2)
econ_params = econ_model.params.drop('const')
econ_params.plot(kind='barh', color='skyblue')
plt.title('Demand Equation: Variable Impact')

plt.tight_layout()
plt.show()
