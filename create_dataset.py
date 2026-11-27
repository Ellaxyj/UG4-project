import geopandas as gpd
import pandas as pd
import numpy as np
from rasterstats import zonal_stats

GADM = "northitaly.geojson"
CASES = "asf_cases_correct.csv"
WILD_BOAR = "D_4.2_A_DensityModel_grid2km_WildBoar_BR_final_MESS_D_pred.gpkg"
POPULATION = "ita_pop_2026_CN_100m_R2025A_v1.tif"
YEAR = 2026
OUTPUT = "dataset_2026.csv"
#read GADM and ASF cases
gdf = gpd.read_file(GADM).to_crs("EPSG:4326")
cases = pd.read_csv(CASES)
cases = cases[cases["Test"] != "Unsuitable"]               
cases["Date_sample"] = pd.to_datetime(cases["Date_sample"])
cases["Year"] = cases["Date_sample"].dt.year         
cases = cases[cases["Year"] == YEAR]  

#let case into point on a map 
points = gpd.GeoDataFrame(cases,geometry=gpd.points_from_xy(cases["Lon"], cases["Lat"]),crs="EPSG:4326")
points = gpd.sjoin(points, gdf[["GID_3", "geometry"]], predicate="within")

gdf["area"] = gdf.to_crs("EPSG:3035").area / 1_000_000

#wild boar
gdf_3035 = gdf.to_crs("EPSG:3035")                  
boar_stats = zonal_stats(gdf_3035, WILD_BOAR, stats="mean", all_touched=True)
gdf["Boar_density"] = pd.DataFrame(boar_stats)["mean"].values


#population
pop_stats = zonal_stats(gdf, POPULATION, stats="sum")   
gdf["Population"] = pd.DataFrame(pop_stats)["sum"].values
gdf["Pop_density"] = gdf["Population"] / gdf["area"]

dataset = points[["GID_3", "Date_sample", "Test"]]

info = gdf[["GID_3", "NAME_1", "NAME_2", "NAME_3", "Boar_density", "Pop_density"]]
dataset = dataset.merge(info, on="GID_3", how="left")

dataset = dataset.rename(columns={
    "NAME_1": "Region",
    "NAME_2": "Province",
    "NAME_3": "Municipality",
    "Date_sample": "Date",
})

dataset = dataset[["GID_3", "Region", "Province", "Municipality",
                   "Date", "Test",
                   "Boar_density", "Pop_density"]]

dataset = dataset.sort_values(["Date", "GID_3"])
dataset.to_csv(OUTPUT, index=False)