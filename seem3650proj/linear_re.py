import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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

# Linear Regression with different test sizes
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
linear_mse = []
for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    y_pred = linear.predict(X_test)
    linear_mse.append(mean_squared_error(y_test, y_pred))

for k in range(8):
    print(f"test size = {(k+1)*0.1:.1f}: MSE = {linear_mse[k]}")
print("Linear Regression with different test sizes: Average MSE = {:.2f}".format(np.mean(linear_mse)))