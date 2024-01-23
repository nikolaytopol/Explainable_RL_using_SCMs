# write a code tthat tests dataset for optimal number of clusters
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
from main import load_dataframe
import sys
import pandas as pd
from sklearn.decomposition import PCA
from helpers.functions.functions import load_dataframe


# "Training_datasets/Bipedal_walker_PPO_training_data.pkl"
"Training_datasets/Pendulum_training_data.pkl"

path="Training_datasets/Bipedal_walker_PPO_training_data.pkl"

if path=="Training_datasets/Pendulum_training_data.pkl":
   dfn=load_dataframe("Training_datasets/Pendulum_training_data.pkl")
   actions_array = dfn['actions'].values
elif path=='Training_datasets/Bipedal_walker_PPO_training_data.pkl':

  dfn=load_dataframe("Training_datasets/Bipedal_walker_PPO_training_data.pkl")
  # Combine columns into a list in a new column
  dfn['action'] = dfn.apply(lambda row: [row['Torque on Hip Joint 1'], row['Torque on Knee Joint 1'], 
                                      row['Torque on Hip Joint 2'], row['Torque on Knee Joint 2']], axis=1)


  # Optional: Drop the original columns if they are no longer needed
  dfn.drop(['Torque on Hip Joint 1', 'Torque on Knee Joint 1', 'Torque on Hip Joint 2', 'Torque on Knee Joint 2'], axis=1, inplace=True)

  # Define the actions data (replace with your actual action data)
  actions_array = dfn['action'].values

# Convert the list of arrays into a two-dimensional NumPy array
actions = np.vstack(actions_array)


# List to store the within-cluster sum of squares for different k values
wcss = []

# Trying different numbers of clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(actions)
    wcss.append(kmeans.inertia_)

print(wcss)
# Plotting the results onto a line graph to observe the 'Elbow'
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within-cluster sum of squares
plt.show()


# Define the number of clusters
num_clusters = 4

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(actions)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print(cluster_labels)

if __name__ == "__main__":
  if path=="Training_datasets/Pendulum_training_data.pkl":
    sys.exit()
   
  # Apply PCA for dimensionality reduction
  pca = PCA(n_components=2)
  reduced_actions = pca.fit_transform(actions)

    # Visualize the clustering result with reduced dimensions
  plt.scatter(reduced_actions[:, 0], reduced_actions[:, 1], c=cluster_labels, cmap='viridis')
  plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.title('K-means Clustering Result (Reduced Dimensions)')

  plt.show()



# 

