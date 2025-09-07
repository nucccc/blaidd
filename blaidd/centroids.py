from typing import Any

import narwhals as nw
import numpy as np
import numpy.typing as npt
import polars as pl
from narwhals.typing import IntoFrame


def calc_centroids(
    df: IntoFrame,
    x_col: str,
    y_col: str,
    label_col: str,
) -> tuple[list[Any], npt.ArrayLike]:
    ndf = nw.from_native(df)
    centroid_rows: list[list[float]] = list()
    labels = list()

    for label, sub_df in ndf.group_by(label_col):
        centroid_row = [sub_df[x_col].mean(), sub_df[y_col].mean()]
        centroid_rows.append(centroid_row)
        labels.append(label[0])
    
    return (
        labels,
        np.array(centroid_rows, dtype=np.float64)
    )
