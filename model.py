import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Data
try:
    df = pd.read_csv('house_prices_dataset.csv')
except FileNotFoundError:
    print("Error: File csv tidak ditemukan.")
    exit()

# 2. Feature Engineering (Sesuai PDF)
def categorize_age(age):
    if age <= 10: return 'New'
    elif age <= 30: return 'Modern'
    else: return 'Old'

df['Age_Category'] = df['age'].apply(categorize_age)

# 3. Split Data
X = df.drop(['price'], axis=1)
y = df['price']

# 4. Setup Pipeline
categorical_features = ['Age_Category']
numerical_features = ['square_feet', 'num_rooms', 'age', 'distance_to_city(km)']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# 5. Train
print("Sedang melatih model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# 6. Evaluasi & SIMPAN SKOR (Bagian Baru)
y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Hasil -> R2: {r2:.4f}, RMSE: {rmse:,.0f}")

# Simpan Model
joblib.dump(model_pipeline, 'model_decision_tree.pkl')

# Simpan Metrics ke file terpisah agar bisa dibaca app.py
metrics = {'r2': r2, 'rmse': rmse}
joblib.dump(metrics, 'model_metrics.pkl') 

print("Selesai! File 'model_decision_tree.pkl' dan 'model_metrics.pkl' berhasil disimpan.")