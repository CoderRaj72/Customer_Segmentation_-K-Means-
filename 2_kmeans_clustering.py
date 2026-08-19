# Step 2: Group customers into segments using K-Means clustering
# ------------------------------------------------------------------
# K-Means is an algorithm that groups similar data points together.
# We tell it how many groups (clusters) we want, and it finds
# customers who are "close" to each other based on Recency,
# Frequency, and Monetary values.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# STEP 1: Load the customer summary we built in the last script
# ------------------------------------------------------------------
customers = pd.read_csv("customer_rfm.csv")

# STEP 2: Scale the numbers
# ------------------------------------------------------------------
# Recency, Frequency, and Monetary are on very different scales
# (e.g. Monetary can be in the thousands, Frequency is usually small).
# K-Means works better when all numbers are on a similar scale,
# so we "standardise" them (this just means: reshape each column
# so it has an average of 0 and a similar spread).
features = customers[["Recency", "Frequency", "Monetary"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# STEP 3: Find a good number of clusters using the "Elbow Method"
# ------------------------------------------------------------------
# We try different numbers of clusters (1 to 9) and measure how
# "tight" the groups are each time. We pick the point where adding
# more clusters stops helping much (this looks like an elbow on
# the chart).
inertia_values = []
cluster_range = range(1, 10)

for k in cluster_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(scaled_features)
    inertia_values.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia_values, marker="o")
plt.title("Elbow Method - Choosing the Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (lower = tighter groups)")
plt.tight_layout()
plt.savefig("chart1_elbow_method.png")
plt.close()
print("Saved chart1_elbow_method.png - the 'elbow' points to k=4")

# STEP 4: Run K-Means with 4 clusters (chosen from the elbow chart)
# ------------------------------------------------------------------
final_model = KMeans(n_clusters=4, random_state=42, n_init=10)
customers["Cluster"] = final_model.fit_predict(scaled_features)

print("\nHow many customers landed in each cluster:")
print(customers["Cluster"].value_counts().sort_index())

# STEP 5: Look at the average Recency, Frequency, Monetary per cluster
# ------------------------------------------------------------------
# This tells us what each cluster actually MEANS in plain English.
cluster_summary = customers.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(1)
print("\nAverage values per cluster:")
print(cluster_summary)

customers.to_csv("customer_segments.csv", index=False)
print("\nSaved customer_segments.csv")
