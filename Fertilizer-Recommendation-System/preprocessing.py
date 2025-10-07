#preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r'C:\Users\DIVYANSHU\Desktop\fer\Fertilizer-Recommendation-System\fertilizer_data.csv')

# Encode categorical columns
le_crop = LabelEncoder()
df['Crop_type'] = le_crop.fit_transform(df['Crop_Type'])

le_fert = LabelEncoder()
df['Recommended_Fertilizer'] = le_fert.fit_transform(df['Recommended_Fertilizer'])

# Features & Target
X = df.drop('Recommended_Fertilizer', axis=1)
y = df['Recommended_Fertilizer']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Preprocessing done!")
