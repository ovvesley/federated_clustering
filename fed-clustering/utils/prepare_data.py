import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

from helpers import get_hash_trial

HASH_trial = get_hash_trial()

def prepare_data(
    input_csv: str,
    output_dir: str,
    columns_to_use: list,
    randomize: bool = False,
    filename: Optional[str] = None,
    file_format="csv",
):
    # Load data from CSV
    df = pd.read_csv(input_csv, header=0, low_memory=False)
    df = df.fillna(-1)
    df = df.apply(pd.to_numeric, errors="coerce")  # Convert non-numeric to NaN

    columns_to_use = ['COADD_OBJECT_ID', 'MAG_G','MAG_R','MAG_I','MAG_Z','MAG_Y','GMR','RMI','IMZ']
    df = df[columns_to_use]

    if randomize:
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    ids = (
        df["COADD_OBJECT_ID"].values
        if "COADD_OBJECT_ID" in df.columns
        else np.arange(len(df))
    )
    # df = df.drop(columns=["coadd_object_id"], errors="ignore").astype(float)

    os.makedirs(output_dir, exist_ok=True)

    filename = filename if filename else "processed_data.csv"
    if file_format == "csv":
        file_path = os.path.join(output_dir, filename)
        ids_df = pd.DataFrame(ids, columns=["COADD_OBJECT_ID"])
        ids_df.to_csv(file_path, sep=",", index=False, header=True)
        df.to_csv(file_path, sep=",", index=False, header=False)
    else:
        raise NotImplementedError("Only CSV format is supported.")


def main():
    parser = argparse.ArgumentParser(description="Read CSV data and process it")
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--columns",
        type=str,
        help="Comma-separated list of columns to use (including derived columns already present in the dataset)",
    )
    parser.add_argument(
        "--randomize", type=int, default=0, help="Whether to randomize data sequence"
    )
    parser.add_argument(
        "--out_path", type=str, required=True, help="Path to output data file"
    )
    args = parser.parse_args()

    from time import perf_counter
    from dfa_lib_python.task import Task
    from dfa_lib_python.dataset import DataSet
    from dfa_lib_python.element import Element

    dataflow_tag = "nvidiaflare-df"

    t1 = Task(1, dataflow_tag, "PrepareData")
    t1.begin()
    start = perf_counter()

    output_dir = os.path.dirname(args.out_path)
    filename = os.path.basename(args.out_path)

    columns_to_use = ['COADD_OBJECT_ID', 'MAG_G','MAG_R','MAG_I','MAG_Z','MAG_Y','GMR','RMI','IMZ']


    # columns_to_use = [col.strip() for col in args.columns.split(",")]

    prepare_data(
        args.input_csv, output_dir, columns_to_use, bool(args.randomize), filename
    )

    duration = perf_counter() - start

    to_dfanalyzer = [
        HASH_trial,
        args.input_csv,
        str(columns_to_use),
        args.randomize,
        args.out_path,
        duration,
    ]
    t1_input = DataSet("iPrepareData", [Element(to_dfanalyzer)])
    t1.add_dataset(t1_input)
    t1_output = DataSet("oPrepareData", [Element([])])
    t1.add_dataset(t1_output)
    t1.end()


if __name__ == "__main__":
    main()
