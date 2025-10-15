# Practical 7: Data Preprocessing and Feature Engineering

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

# ------------------------------
# 1️⃣ Create a sample dataset with missing and categorical values
# ------------------------------
data = {
    'Age': [25, np.nan, 30, 22, np.nan, 28],
    'Salary': [50000, 54000, np.nan, 58000, 60000, np.nan],
    'Gender': ['Male', 'Female', 'Female', 'Male', np.nan, 'Female'],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Chennai', 'Mumbai', 'Delhi']
}

df = pd.DataFrame(data)
print("🔹 Original Dataset:")
print(df)

# ------------------------------
# 2️⃣ Handle Missing Data using SimpleImputer
# ------------------------------
imputer = SimpleImputer(strategy='mean')  # for numerical columns
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# Fill missing categorical values with most frequent
imputer_cat = SimpleImputer(strategy='most_frequent')
df[['Gender']] = imputer_cat.fit_transform(df[['Gender']])

print("\n🔹 After Handling Missing Data:")
print(df)

# ------------------------------
# 3️⃣ Encode Categorical Variables
# ------------------------------

# Label Encoding for 'Gender' (binary category)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# OneHot Encoding for 'City' (multiple categories)
df_encoded = pd.get_dummies(df, columns=['City'], drop_first=True)

print("\n🔹 After Encoding Categorical Variables:")
print(df_encoded)

# ------------------------------
# 4️⃣ Feature Scaling using StandardScaler and MinMaxScaler
# ------------------------------
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

# Apply scaling only on numerical columns
numeric_cols = ['Age', 'Salary']
df_encoded[numeric_cols] = scaler_standard.fit_transform(df_encoded[numeric_cols])

print("\n🔹 After Feature Scaling (StandardScaler):")
print(df_encoded)

# ------------------------------
# 5️⃣ Split the Dataset into Training and Testing Sets
# ------------------------------
X = df_encoded.drop('Gender', axis=1)
y = df_encoded['Gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n🔹 Training Set:")
print(X_train)
print("\n🔹 Testing Set:")
print(X_test)
