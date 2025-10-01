# Import necessary libraries
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

"""
Vehicle Clustering (Unsupervised Learning)
Scenario - You are working for an automotive company, and your task is to cluster vehicles into groups based on their features such as weight, engine size, and horsepower.
"""

# Initialize the random number generator
rng = np.random.default_rng(0)

# Generate synthetic dataset for vehicles using the new random generator
data_size = 300
data = {
    'Weight': rng.integers(1000, 3000, data_size),
    'EngineSize': rng.uniform(1.0, 4.0, data_size),
    'Horsepower': rng.integers(50, 300, data_size),
}
df = pd.DataFrame(data)

# No labels are needed for unsupervised learning
X = df

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Plotting the clusters
plt.scatter(df['Weight'], df['Horsepower'], c=kmeans.labels_)
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title('Vehicle Clusters')
plt.show()
