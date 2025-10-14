import pandas as pd

# ---------------------------
# STEP 1: Load Dataset
# ---------------------------
data = pd.read_csv('data.csv')
print("✅ Dataset Loaded Successfully!\n")
print(data)

# ---------------------------
# STEP 2: Handle Missing Values
# ---------------------------
print("\n🔍 Missing Values:\n", data.isnull().sum())

# Fill or drop missing values
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['City'].fillna('Unknown', inplace=True)
data.dropna(subset=['Salary'], inplace=True)

print("\n✅ After Handling Missing Values:\n", data)

# ---------------------------
# STEP 3: Remove Duplicates
# ---------------------------
data.drop_duplicates(inplace=True)
print("\n🧹 After Removing Duplicates:\n", data)

# ---------------------------
# STEP 4: Analyze Data
# ---------------------------
# Group by City and calculate average salary
avg_salary = data.groupby('City')['Salary'].mean()
print("\n📊 Average Salary by City:\n", avg_salary)

# Describe statistics
print("\n📈 Statistical Summary:\n", data.describe())

# Filter records where Salary > 55,000
high_salary = data[data['Salary'] > 55000]
print("\n💼 Employees with Salary > 55,000:\n", high_salary)
