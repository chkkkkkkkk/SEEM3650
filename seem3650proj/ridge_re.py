import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load the data
districts = []
average_speed = []
population_density = []
with open("output.txt", "r") as f:
    lines = f.readlines()
    districts = [x.strip() for x in lines[0].split(",")]
    average_speeds = [float(x) for x in lines[1].split(",")]
    population_density = [int(x) for x in lines[2].split(",")]

# Create a dataframe
df = pd.DataFrame({'District': districts, 'Average Speed': average_speeds, 'Population Density': population_density})

# Define the features and target variable
X = df[['Population Density']]
y = df['Average Speed']

# Ridge Regression with cross-validation
ridge = Ridge(alpha=1.0)
kf = KFold(n_splits=5, shuffle=True)
ridge_mse = []
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_val)
    ridge_mse.append(mean_squared_error(y_val, y_pred))

for k in range(5):
    print(f"test size = {k+1}: MSE = {ridge_mse[k]}")
print("Ridge Regression with cross-validation: Average MSE = {:.2f}".format(np.mean(ridge_mse)))