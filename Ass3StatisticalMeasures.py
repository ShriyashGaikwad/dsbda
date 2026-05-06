import pandas as pd
import numpy as np

df = pd.read_csv("iris.csv")
df.head()

stats = df.groupby("Species")["SepalLengthCm"].agg(
    Mean="mean",
    Median="median",
    Minimum="min",
    Maximum="max",
    Std_Deviation="std"
)

print(stats)

stats["Mean"].plot(kind="bar")

df.boxplot(column="SepalLengthCm", by="Species")

for sp in df["Species"].unique():
    print("\nStatistics for", sp)
    print(df[df["Species"]==sp][
        ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
    ].describe())