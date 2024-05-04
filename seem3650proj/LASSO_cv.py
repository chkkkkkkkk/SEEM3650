import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
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

# LASSO Regression with k-fold cross-validation
lasso = Lasso(alpha=0.1)
kf = KFold(n_splits=5, shuffle=True)
lasso_mse = []
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_val)
    lasso_mse.append(mean_squared_error(y_val, y_pred))
for k in range(5):
    print(f"k = {k+1}: MSE = {lasso_mse[k]}")
print("LASSO Regression with k-fold cross-validation: Average MSE = {:.2f}".format(np.mean(lasso_mse)))