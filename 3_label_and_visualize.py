# Step 3: Give each cluster a plain-English name and visualise them
# ------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

customers = pd.read_csv("customer_segments.csv")

# Looking at the average Recency/Frequency/Monetary for each cluster
# (from the last script's output), we can describe them in plain
# English like this:
#
#   Cluster 0: Medium recency, medium frequency, medium spend
#              -> "Regular Customers"
#   Cluster 1: High recency (long time since last order), low
#              frequency, low spend -> "At-Risk / Inactive Customers"
#   Cluster 2: Very recent, extremely frequent, extremely high spend
#              -> "Best Customers (VIP)"
#   Cluster 3: Recent, frequent, high spend (but not as extreme as
#              Cluster 2) -> "Loyal High-Value Customers"

cluster_names = {
    0: "Regular Customers",
    1: "At-Risk / Inactive Customers",
    2: "Best Customers (VIP)",
    3: "Loyal High-Value Customers",
}

customers["Segment"] = customers["Cluster"].map(cluster_names)

print("Number of customers in each segment:")
print(customers["Segment"].value_counts())

# Save the final labelled table
customers.to_csv("customer_segments_labelled.csv", index=False)
print("\nSaved customer_segments_labelled.csv")

# STEP: Make a simple chart - Frequency vs Monetary, coloured by segment
# ------------------------------------------------------------------
# We use a log scale on Monetary because a few customers spend WAY
# more than everyone else, which would otherwise squash the chart.
plt.figure(figsize=(9, 6))
colors = {"Regular Customers": "#5b9bd5",
          "At-Risk / Inactive Customers": "#ed7d31",
          "Best Customers (VIP)": "#70ad47",
          "Loyal High-Value Customers": "#7030a0"}

for segment, group in customers.groupby("Segment"):
    plt.scatter(group["Frequency"], group["Monetary"],
                label=segment, alpha=0.6, color=colors[segment])

plt.yscale("log")
plt.xlabel("Frequency (number of orders)")
plt.ylabel("Monetary (total spend, £ - log scale)")
plt.title("Customer Segments: Frequency vs Spend")
plt.legend()
plt.tight_layout()
plt.savefig("chart2_customer_segments.png")
plt.close()
print("Saved chart2_customer_segments.png")

# STEP: Bar chart - how many customers are in each segment
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
customers["Segment"].value_counts().plot(kind="bar", color="steelblue")
plt.title("Number of Customers per Segment")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("chart3_segment_sizes.png")
plt.close()
print("Saved chart3_segment_sizes.png")
