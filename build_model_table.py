import pandas as pd
import numpy as np

data = pd.read_csv("dataset_all_years.csv")
data["Date"] = pd.to_datetime(data["Date"])

#separate the half year
year = data["Date"].dt.year.astype(str)
half = np.where(data["Date"].dt.month <= 6, "-01_to_" + year + "-06", "-07_to_" + year + "-12")
data["Period"] = year + half

#create a new column to indicate if the test is positive or not
data["is_positive"] = 0
data.loc[data["Test"] == "Positive", "is_positive"] = 1

# group by GID_3 and Period, and aggregate the data
table = data.groupby(["GID_3", "Period"]).agg(
    Region=("Region", "first"),
    Province=("Province", "first"),
    Municipality=("Municipality", "first"),
    Samples=("is_positive", "count"),
    Positive=("is_positive", "sum"),
    Boar_density=("Boar_density", "first"),
    Pop_density=("Pop_density", "first"),
).reset_index()

#label the positive and negative municipalities
table["label"] = 0
table.loc[table["Positive"] > 0, "label"] = 1

#sort the columns and the rows
table = table[["GID_3", "Region", "Province", "Municipality", "Period",
               "Samples", "Positive", "label",
               "Boar_density", "Pop_density"]]
table = table.sort_values(["Period", "GID_3"])
table.to_csv("model_table.csv", index=False)
