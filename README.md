# Customer_Segmentation_-K-Means-

## What this project does
Groups real customers of an online retail store into segments based
on their buying behaviour, so the business can treat different types
of customers differently (e.g. reward loyal customers, try to win
back customers who disappeared).

## Dataset
**Name:** Online Retail Dataset (same dataset as Project 1)
**Description:** Real transactions from a UK-based online gift shop,
01 Dec 2010 to 09 Dec 2011 (541,909 rows, 4,338 unique customers).
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail), donated by Dr. Daqing Chen, London South Bank University.
**File used in this project:** `retail_data.csv`

## The core idea: RFM
You can't cluster raw transaction rows directly (each row is just
one item bought). So first, each customer is summarised into 3 simple
numbers, called RFM:
- **Recency** — days since their last order (lower = better, means they bought recently)
- **Frequency** — how many separate orders they've placed (higher = better)
- **Monetary** — how much money they've spent in total (higher = better)

Then K-Means clustering groups customers who have similar RFM values.

## Files in this project
- `1_build_rfm.py` — turns raw transactions into one row per customer (RFM)
- `2_kmeans_clustering.py` — scales the data, finds the best number of clusters (Elbow Method), and runs K-Means
- `3_label_and_visualize.py` — gives each cluster a plain-English name and creates charts
- `retail_data.csv` — the raw dataset
- `customer_rfm.csv` — one row per customer with Recency, Frequency, Monetary
- `customer_segments_labelled.csv` — final table with each customer's segment name
- `chart1_elbow_method.png` — chart used to choose the number of clusters
- `chart2_customer_segments.png` — scatter plot showing the 4 segments
- `chart3_segment_sizes.png` — how many customers are in each segment

## How to run it
1. Install the needed packages:
   ```
   pip install pandas scikit-learn matplotlib
   ```
2. Run the files in order:
   ```
   python 1_build_rfm.py
   python 2_kmeans_clustering.py
   python 3_label_and_visualize.py
   ```

## Key findings: the 4 customer segments found

| Segment | Customers | What they're like |
|---|---|---|
| **Regular Customers** | 3,054 (70%) | Buy occasionally, moderate spend — the bulk of the customer base |
| **At-Risk / Inactive Customers** | 1,067 (25%) | Haven't ordered in a long time (~248 days on average), low spend — likely to be lost |
| **Loyal High-Value Customers** | 204 (5%) | Order often, spend well (~£12,700 total) — a smaller but very valuable group |
| **Best Customers (VIP)** | 13 (<1%) | Extremely frequent orders and huge total spend (~£127,000 average) — almost certainly wholesale buyers, not typical individual shoppers |

## What this means for the business
- The **At-Risk segment (25% of customers)** is the biggest opportunity —
  a "we miss you" discount campaign could win some of them back before
  they're lost for good.
- The **13 VIP customers** contribute a huge, disproportionate amount of
  revenue and should be given dedicated account management, since
  losing even one of them would hurt significantly.
- The **Loyal High-Value group (204 customers)** are good candidates
  for a loyalty or rewards programme to keep them engaged.
