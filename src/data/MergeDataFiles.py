import numpy as np
import pandas as pd

from pathlib import Path
from zipfile import ZipFile

COLUMNS = ["Time", "Hacc", "Vacc", "Bearing"]

# change directory to raw files here and merged files at the end
ROOT_FROM = Path("../../data/raw/Test_set/")
ROOT_TO = Path("../../data/MergedTest_files/")

ROOT_FROM.mkdir(parents=True, exist_ok=True)
ROOT_TO.mkdir(parents=True, exist_ok=True)


def read_bearings_zip(input_zip):
    """
    Extract the bearings Test_set.zip file into memory.

    Parameters
    ----------
    input_zip : Path or str
        Path to the Test_set.zip file.

    Returns
    -------
    files : dict of dicts
        Dict of dicts of contained .csv files as pandas dataframes.
    """

    input_zip = ZipFile(input_zip)
    files = {}
    for name in sorted(input_zip.namelist()):
        # Add 'directory' in dict if not yet present
        parent = Path(name).parent.stem
        if parent not in files:
            if "Bearing" not in parent:
                continue
            files[parent] = {}

        # Read and store .csv contents
        with input_zip.open(name) as file:
            if "acc" in name:
                files[parent][name] = pd.read_csv(
                    file,
                    header=None,
                    names=["h", "m", "s", "ms", "Hacc", "Vacc"],
                    dtype={"ms": np.int32},
                    sep=";"
                    if "1.4" in parent
                    else ",",  # 1.4 uses ; to separate values
                )
    return files


# Read the zip file into memory
input_data = read_bearings_zip(Path("../../data/raw/Test_set.zip"))


for dir, files in input_data.items():
    print("Directory:", dir)

    # Concatenate individual .csv files
    tdf = pd.concat(files.values(), ignore_index=True)

    # Convert individual time columns to a single time column
    tdf["Time"] = pd.to_datetime(
        tdf["h"] * 3600000 + tdf["m"] * 60000 + tdf["s"] * 1000 + tdf["ms"] / 1000,
        unit="ms",
    ).dt.time
    tdf["Bearing"] = dir

    # Save to csv (slow)
    tdf[COLUMNS].to_csv(ROOT_TO / f"{dir}.csv", index=False)
