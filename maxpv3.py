import geopandas as gpd
import libpysal
from spopt.region import MaxPHeuristic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy

random_seed = 123456

# Step 1: Load the shapefile
shapefile_path = "SPOPT_DATA_JOINED.shp"
gdf = gpd.read_file(shapefile_path)

# Step 2: Convert 'spopt_v244' to numeric and handle missing values
gdf["spopt_v244"] = pd.to_numeric(gdf["spopt_v244"], errors="coerce")
gdf["spopt_v244"] = gdf["spopt_v244"].fillna(gdf["spopt_v244"].median())

# Step 3: Create a spatial weights matrix using Queen contiguity
w = libpysal.weights.Queen.from_dataframe(gdf)


# Step 4: Define the attribute names and threshold parameters
attrs_name = ["spopt_v244"]
gdf["count"] = 1
threshold_name = "count"
# Increase threshold and top_n for larger regions
threshold = 40  # Increase this value
top_n = 10       # Consider more candidate regions

# Step 5: Set the random seed for reproducibility
np.random.seed(random_seed)

# Step 6: Initialize and solve the Max-P model
model = MaxPHeuristic(gdf, w, attrs_name, threshold_name, threshold, top_n)
model.solve()

# Step 7: Assign the region labels back to the GeoDataFrame
gdf["region"] = model.labels_

# Step 8: Print the resulting data with region labels
print(gdf[["spopt_v244", "region"]].head())

# Create a histogram of the resulting clusters (region labels)
plt.figure(figsize=(10, 6))
gdf["region"].value_counts().sort_index().plot(kind="bar", color="skyblue", edgecolor="black")
plt.xlabel("Region Label")
plt.ylabel("Number of Census Tracts")
plt.title("Distribution of Census Tracts Across Resulting Clusters")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.savefig("cluster_histogram.png", dpi=300, bbox_inches="tight")
plt.show()


# Step 9: Visualize the resulting regions
gdf.plot(column="region", legend=True, cmap="Set3", edgecolor="black")
plt.title("Max-P Clustering of Census Tracts by Income per Capita")
plt.savefig("maxp_map.png", dpi=300, bbox_inches="tight")
plt.show()


#Step 10: create table showing income per capita distribution across clusters
import pandas as pd

# Calculate the average income per capita for each cluster group
cluster_summary = gdf.groupby("region").agg(
    average_income_per_capita=("spopt_v244", "mean"),
    number_of_census_tracts=("region", "count")
).reset_index()


cluster_summary.columns = ["Region", "Average Income per Capita", "Number of Census Tracts"]

# Display the summary table
print(cluster_summary)

#create shapefile of poorest cluster
poorest_region_label = cluster_summary.loc[cluster_summary["Average Income per Capita"].idxmin(), "Region"]
poorest_region_gdf = gdf[gdf["region"] == poorest_region_label]
output_shapefile_path = "poorest.shp"
poorest_region_gdf.to_file(output_shapefile_path)


