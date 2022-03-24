import numpy as np
import os
import pandas as pd

from pathlib import Path

COLUMNS = ["Time", "Hacc", "Vacc", "Bearing"]

# change directory to raw files here and merged files at the end
ROOT_FROM = Path("../../data/raw/Test_set/")
ROOT_TO = Path("../../data/MergedTest_files/")

ROOT_FROM.mkdir(parents=True, exist_ok=True)
ROOT_TO.mkdir(parents=True, exist_ok=True)

# Go over all the files/directories
for path, dirs, files in os.walk(ROOT_FROM):
    for dir in dirs:
        print(dir)
        bdfs = []
        filenames = os.listdir(os.path.join(path, dir))
        for file in filenames:
            if "acc" in file:
                tdf = pd.read_csv(
                    ROOT_FROM / dir / file,
                    header=None,
                    names=["h", "m", "s", "ms", "Hacc", "Vacc"],
                    dtype={"ms": np.int32},
                    sep=";" if "1.4" in dir else ",",  # 1.4 uses ; to separate values
                )
                bdfs.append(tdf)
        tdf = pd.concat(bdfs, ignore_index=True)

        tdf["Time"] = pd.to_datetime(
            tdf["h"] * 3600000 + tdf["m"] * 60000 + tdf["s"] * 1000 + tdf["ms"] / 1000,
            unit="ms",
        ).dt.time
        tdf["Bearing"] = dir
        tdf[COLUMNS].to_csv(ROOT_TO / f"{dir}.csv", index=False)
else:
    if "bdfs" not in globals():
        print(
            "Dataset not found! Please add the dataset to the data/raw/Test_set folder."
        )
