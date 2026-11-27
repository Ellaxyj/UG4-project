import geopandas as gpd
import pandas as pd
from geotessera import GeoTesseraZarr

GADM = "northitaly.geojson"
YEAR = 2022                
N_POINTS = 30                
OUTPUT = "tessera_" + str(YEAR) + ".csv"

# read the municipalities to be processed
gdf = gpd.read_file(GADM)
model_table = pd.read_csv("model_table.csv")
sampled_ids = model_table["GID_3"].unique()
gdf = gdf[gdf["GID_3"].isin(sampled_ids)]


# sample N_POINTS points in each municipality

gdf_3035 = gdf.to_crs("EPSG:3035").set_index("GID_3")
points = gdf_3035.sample_points(N_POINTS, rng=42)
points = points.explode().to_crs("EPSG:4326")

coords = []
for p in points:
    coords.append((p.x, p.y))          

# read the TESSERA embedding for each point
gt = GeoTesseraZarr()
vectors = gt.sample_points(coords, year=YEAR)


# mean the embedding for each municipality
emb = pd.DataFrame(vectors)
emb.columns = ["tess_" + str(i) for i in range(vectors.shape[1])]
emb["GID_3"] = points.index.values
#remove the rows with NaN values
emb = emb.dropna()                                 
emb_mean = emb.groupby("GID_3",sort=False).mean().reset_index()
emb_mean.to_csv(OUTPUT, index=False)
