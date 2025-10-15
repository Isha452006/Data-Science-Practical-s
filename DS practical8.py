import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dataset
data = {
    'Hours_Studied': [1,2,3,4,5,6,7,8,9,10],
    'Score': [35,40,50,55,60,70,75,80,85,95]
}
df = pd.DataFrame(data)

X = df[['Hours_Studied']]
y = df['Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Polynomial Regression
poly_feat = PolynomialFeatures(degree=2)
X_poly = poly_feat.fit_transform(X_train)

# Make sure X_test has same column names
X_test_df = pd.DataFrame(X_test, columns=X_train.columns)
X_test_poly = poly_feat.transform(X_test_df)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

# Combined Plots
plt.figure(figsize=(15,5))

# Linear Regression Line
plt.subplot(1,3,1)
plt.scatter(X, y, color='lightblue', label='Actual Data')
plt.plot(X, lin_reg.predict(X), color='orange', label='Linear Regression')
plt.title('Linear Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

# Polynomial Regression Curve
plt.subplot(1,3,2)
plt.scatter(X, y, color='lightgreen', label='Actual Data')
plt.plot(X, poly_reg.predict(poly_feat.transform(X)), color='red', label='Polynomial Regression')
plt.title('Polynomial Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

# Residual Errors
plt.subplot(1,3,3)
plt.scatter(y_test, y_test - y_pred_lin, color='violet', label='Linear Residuals')
plt.scatter(y_test, y_test - y_pred_poly, color='pink', label='Polynomial Residuals')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title('Residual Errors')
plt.xlabel('Actual Score')
plt.ylabel('Residual')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.suptitle('Regression Analysis Combined Charts', fontsize=14, fontweight='bold', y=1.05)
plt.show()

# ------------------------------
# Evaluation Metrics
# ------------------------------
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {model_name} Evaluation Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

evaluate_model(y_test, y_pred_lin, "Linear Regression")
evaluate_model(y_test, y_pred_poly, "Polynomial Regression")
