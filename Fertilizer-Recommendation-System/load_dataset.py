import pandas as pd

# Load dataset
df = pd.read_csv(r'C:\Users\Aditya\OneDrive\Desktop\fertilizer_data.csv')

# Show first few rows
print("\n--- First 5 Rows ---")
print(df.head())


# Info about dataset
print("\n--- Info ---")
print(df.info())

# Statistical summary
print("\n--- Description ---")
print(df.describe())

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv(r'C:\Users\DIVYANSHU\Desktop\fer\Fertilizer-Recommendation-System\fertilizer_data.csv')

print("\n--- Before Preprocessing ---")
print(df.head())

# 1. Handle Missing Values
df = df.drop_duplicates()       # remove duplicates
df = df.dropna()                # drop missing rows (if any)

# 2. Encode Categorical Variables
label_encoders = {}
for col in ['Soil_Type', 'Crop_Type', 'Recommended_Fertilizer']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3. Normalize Numerical Features (N, P, K)
scaler = StandardScaler()
df[['N','P','K']] = scaler.fit_transform(df[['N','P','K']])

print("\n--- After Preprocessing ---")
print(df.head())

# Save processed dataset
df.to_csv("fertilizer_data_processed.csv", index=False)
print("\n✅ Preprocessed data saved as fertilizer_data_processed.csv")
