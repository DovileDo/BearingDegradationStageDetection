import pandas as pd
import numpy as np
import os

column_names = ["Time", "Hacc", "Vacc", "Bearing"]

"""change directory to raw files here and merged files at the end"""
rootdir = "../../data/raw/Test_set/"

for subdir, dirs, files in os.walk(rootdir):
    for dir in dirs:
        bdf = pd.DataFrame(columns=column_names)
        filenames = os.listdir(os.path.join(subdir, dir))
        for file in filenames:
            print(file)
            if "acc" in file:
                print(rootdir + "/" + dir + "/" + file)
                tdf = pd.read_csv(
                    rootdir + "/" + dir + "/" + file,
                    # sep=';',
                    header=None,
                    names=["h", "m", "s", "ms", "Hacc", "Vacc"],
                    dtype={"ms": np.int32},
                )

                tdf["Time"] = pd.to_datetime(
                    tdf["h"].astype(str)
                    + " "
                    + tdf["m"].astype(str)
                    + " "
                    + tdf["s"].astype(str)
                    + " "
                    + tdf["ms"].astype(str),
                    format="%H %M %S %f",
                ).dt.time
                tdf["Bearing"] = dir
                tdf = tdf.drop(["h", "m", "s", "ms"], axis=1)
                bdf = bdf.append(tdf, ignore_index=True)
        bdf.to_csv("../../data/MergedTest_files/" + dir + ".csv", index=False)
