import pandas as pd

# files = ["dataset_2022.csv", "dataset_2023.csv", "dataset_2024.csv",
#          "dataset_2025.csv", "dataset_2026.csv"]

# tables = []
# for f in files:
#     table = pd.read_csv(f)
#     tables.append(table)

# all_years = pd.concat(tables)
# all_years = all_years.sort_values(["Date", "GID_3"])
# all_years.to_csv("dataset_all_years.csv", index=False)


#merge the model table with TESSERA embedding
table = pd.read_csv("model_table.csv")
table["Year"] = table["Period"].str[:4].astype(int)

parts = []
for year in [2022, 2023, 2024, 2025]:
    t = pd.read_csv("tessera_" + str(year) + ".csv")
    t["Year"] = year
    parts.append(t)
all_tessera = pd.concat(parts)

# 按 市镇 + 年份 合并：2022 年的行配 2022 年的，2023 年的配 2023 年的……
table = table.merge(all_tessera, on=["GID_3", "Year"], how="left")

print("行数:", len(table))
print("没有 embedding 的行:", table["tess_0"].isna().sum())

table.to_csv("model_table_tessera.csv", index=False)
print("已保存 model_table_tessera.csv")
