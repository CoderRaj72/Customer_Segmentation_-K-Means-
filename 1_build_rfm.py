# Project 3: Customer Segmentation Analysis (K-Means Clustering)
# ------------------------------------------------------------------
# Goal: Group customers into segments based on their buying behaviour,
# so a business can treat different types of customers differently
# (e.g. reward loyal customers, win back customers who disappeared).
#
# Data: Real transactions from a UK-based online retail store
# (same dataset used in Project 1).
#
# The big idea: We can't cluster raw transactions (each row is just
# one item bought). Instead, we first summarise EACH CUSTOMER into
# 3 simple numbers. This is called RFM:
#   Recency   = How many days ago did they last buy something?
#               (small number = bought recently = good)
#   Frequency = How many separate orders have they placed?
#               (big number = orders often = good)
#   Monetary  = How much money have they spent in total?
#               (big number = spends a lot = good)

import pandas as pd

# STEP 1: Load and clean the data (same cleaning as Project 1)
# ------------------------------------------------------------------
data = pd.read_csv("retail_data.csv")
data = data.dropna(subset=["CustomerID"])
data = data[data["Quantity"] > 0]
data = data[data["UnitPrice"] > 0]
data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

# STEP 2: Pick a reference date (1 day after the last purchase in the data)
# ------------------------------------------------------------------
# We measure "Recency" as: reference_date - date of last purchase
reference_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)
print("Reference date used for Recency:", reference_date)

# STEP 3: Build one row per customer with Recency, Frequency, Monetary
# ------------------------------------------------------------------
customer_summary = data.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum")
).reset_index()

print("\nNumber of unique customers:", customer_summary.shape[0])
print("\nFirst 5 customers:")
print(customer_summary.head())

print("\nSummary statistics:")
print(customer_summary[["Recency", "Frequency", "Monetary"]].describe())

# Save this table so the next script can use it
customer_summary.to_csv("customer_rfm.csv", index=False)
print("\nSaved customer_rfm.csv")
