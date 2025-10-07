# model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv(r'C:\Users\DIVYANSHU\Desktop\fer\Fertilizer-Recommendation-System\fertilizer_data.csv')

# 2. Encode categorical columns
le_soil = LabelEncoder()
le_crop = LabelEncoder()

df['Soil_Type'] = le_soil.fit_transform(df['Soil_Type'])   # e.g., Sandy -> 2, Loamy -> 1, Clay -> 0
df['Crop_Type'] = le_crop.fit_transform(df['Crop_Type'])   # Crop names to numbers

# 3. Features (X) and target (y)
X = df.drop(columns=['Recommended_Fertilizer'])  # Fertilizer is our prediction target
y = df['Recommended_Fertilizer']

# 4. Encode Fertilizer target also (because it's string labels)
le_fert = LabelEncoder()
y = le_fert.fit_transform(y)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Save model and encoders
joblib.dump(model, "fertilizer_model.pkl")
joblib.dump(le_soil, "soil_encoder.pkl")
joblib.dump(le_crop, "crop_encoder.pkl")
joblib.dump(le_fert, "fertilizer_encoder.pkl")

print("✅ Model training complete. Model and encoders saved!")
