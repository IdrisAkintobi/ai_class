# Import necessary libraries
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

"""
Predicting Building Energy Efficiency (Supervised Learning)
Scenario - You are working for an architecture firm, and your task is to build a model 
that predicts the energy efficiency rating of buildings based on features like 
wall area, roof area, overall height, etc.
"""

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Generate synthetic dataset for building features and energy efficiency ratings
# Initialize the random number generator
rng = np.random.default_rng(0)

# Define the size of the data
data_size = 500

# Create the dataset
data = {
    'WallArea': rng.integers(200, 400, size=data_size),
    'RoofArea': rng.integers(100, 200, size=data_size),
    'OverallHeight': rng.uniform(3, 10, size=data_size),
    'GlazingArea': rng.uniform(0, 1, size=data_size),
    'EnergyEfficiency': rng.uniform(10, 50, size=data_size),  # Energy efficiency rating
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Data preprocessing
X = df.drop('EnergyEfficiency', axis=1)
y = df['EnergyEfficiency']

# Visualize the relationships between features and the target variable (Energy Efficiency)
sns.pairplot(
    df,
    x_vars=['WallArea', 'RoofArea', 'OverallHeight', 'GlazingArea'],
    y_vars='EnergyEfficiency',
    height=4,
    aspect=1,
    kind='scatter',
)
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model with specified hyperparameters
model = RandomForestRegressor(min_samples_leaf=5, max_features='sqrt', random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot the True values vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()
