# Step one: Load libraries
import geopandas as gpd
import pandas as pd
import libpysal
import matplotlib.pyplot as plt
import numpy
import pandas
import shapely
from sklearn.metrics import pairwise as skm
import spopt
import warnings


# Step two: load + clean + project dataset 
shapefile_path = "SPOPT_DATA_JOINED.shp"
gdf = gpd.read_file(shapefile_path)
gdf["spopt_v244"] = pd.to_numeric(gdf["spopt_v244"], errors="coerce")
gdf["spopt_v244"] = gdf["spopt_v244"].fillna(gdf["spopt_v244"].median())
print(gdf.crs)
gdf = gdf.to_crs("EPSG:2272")

#Step three: define parameters for SKATER
attrs_name = ["spopt_v244"] #variable we are using to regionalize
w = libpysal.weights.Queen.from_dataframe(gdf) # create spatial weights object from shp
n_clusters = 12 # number of contigous regions we want 
floor = 1 # minimum number of spatial objects in each region 
trace = False #wether or not we store intermediate values
islands = "increase"

#Step four: create spanning forest (MST)
spanning_forest_kwds = dict(
    dissimilarity=skm.manhattan_distances,
    affinity=None,
    reduction=numpy.sum,
    center=numpy.mean,
    verbose=2
)

#Step four: solve the model 
model = spopt.region.Skater(
    gdf,
    w,
    attrs_name,
    n_clusters=n_clusters,
    floor=floor,
    trace=trace,
    islands=islands,
    spanning_forest_kwds=spanning_forest_kwds
)
model.solve()

#Add to data frame
gdf["demo_regions"] = model.labels_

#visualize 
gdf["region"] = model.labels_  # Assign the cluster labels to the GeoDataFrame
gdf.plot(column="region", categorical=True, legend=True, figsize=(10, 6))
plt.title("Skater Clustering with Islands as Separate Regions")
plt.show()